"""Putting N aligned models into one set of weights.

Two merge rules live here:

``naive_average``
    The floor.  Mean of the weights, no alignment, no disentangling.  Every
    result in this repo is reported against it.

``fuse``
    The full pipeline: canonicalize (Phase 1), measure effective ranks and
    disentangle the used subspaces (Phase 2), then combine.  When
    ``sum_k r_k <= d`` the models occupy orthogonal directions and the
    combination is *literally a sum* with no interference; over capacity, the
    budget QP picks what to keep and reports the captured energy ``rho``.

The distinction between sum and mean is not cosmetic.  Averaging N models that
occupy disjoint subspaces attenuates every model's contribution by ``1/N``,
which the downstream RMSNorm partly but not entirely undoes.  When the
subspaces really are disjoint, summing is the correct combination, and the
capacity condition is exactly the condition under which summing is safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..config import FusionConfig
from ..model import Transformer, require_same_arch
from ..symmetry import apply_residual_rotation
from .rank import RankProfile, participation_ratio, read_point_covariance, used_subspace
from .stiefel import (
    KNEE,
    DisentangleResult,
    budget_qp,
    disentangle,
    interference_ratio,
    rearrangement_floor,
    subspace_overlap,
)


@dataclass
class CapacityReport:
    """Everything needed to attribute a merge outcome to a phase."""

    profiles: list[RankProfile] = field(default_factory=list)
    total_rank: int = 0
    d_model: int = 0
    overlap_before: float = float("nan")
    overlap_after: float = float("nan")
    feasible: bool = False
    captured_energy: float = float("nan")
    participation: list[float] = field(default_factory=list)
    rank_energy: list[float] = field(default_factory=list)
    disentangled: bool = False
    combine: str = ""
    weighted: bool = False
    interference_before: float = float("nan")
    interference_after: float = float("nan")
    floor: float = float("nan")
    at_floor: bool = False
    #: The disentangling rotation applied to each model before summing.
    #: Needed to compare the merged model against a specialist: after fusion the
    #: two live in different frames, and a residual-stream comparison across
    #: frames measures nothing.
    rotations: list[Tensor] = field(default_factory=list)
    #: The models as they were summed -- canonicalized and rotated into the
    #: merged frame. These are what a per-model diagnostic must be run against.
    aligned: list[Transformer] = field(default_factory=list)

    @property
    def rank_coverage(self) -> float:
        """Worst per-model energy coverage of the bases the capacity test used."""
        return min(self.rank_energy) if self.rank_energy else float("nan")

    @property
    def ratio(self) -> float:
        return self.total_rank / self.d_model if self.d_model else float("nan")

    def __str__(self) -> str:
        if self.weighted:
            verdict = "below knee" if self.interference_after < KNEE else "ABOVE KNEE"
            stalled = "" if self.at_floor else "  [above floor: solver, not capacity, is the limit]"
            return (
                f"weighted overlap {self.overlap_before:.4f} -> {self.overlap_after:.4f} "
                f"(floor {self.floor:.4f}); interference {self.interference_before:.3f} -> "
                f"{self.interference_after:.3f} ({verdict}); "
                f"combine={self.combine}{stalled}"
            )
        head = (
            f"sum(r_k)/d = {self.total_rank}/{self.d_model} = {self.ratio:.2f} "
            f"({'feasible' if self.feasible else 'OVER CAPACITY'})"
        )
        mid = f"overlap {self.overlap_before:.4f} -> {self.overlap_after:.4f}"
        tail = "" if self.feasible else f", captured energy rho = {self.captured_energy:.3f}"
        out = f"{head}; {mid}; combine={self.combine}{tail}"
        warnings = [w for p in self.profiles if (w := p.warn_if_low_energy())]
        if warnings:
            out += f"\n    CAVEAT: {warnings[0]}"
        return out


@torch.no_grad()
def naive_average(models: list[Transformer]) -> Transformer:
    """Mean of the raw weights. The floor, and the thing to beat."""
    require_same_arch(*models)
    out = models[0].clone()
    device = next(out.parameters()).device
    named = [dict(m.named_parameters()) for m in models]
    for name, p in out.named_parameters():
        p.data = torch.stack([n[name].data.to(device) for n in named]).mean(0).to(p.dtype)
    return out


@torch.no_grad()
def average_models(models: list[Transformer], weights: list[float] | None = None) -> Transformer:
    """Weighted mean, applied to models assumed to be already aligned."""
    require_same_arch(*models)
    out = models[0].clone()
    device = next(out.parameters()).device
    w = torch.tensor(
        weights if weights else [1.0 / len(models)] * len(models),
        dtype=torch.float64, device=device,
    )
    w = w / w.sum()
    named = [dict(m.named_parameters()) for m in models]
    for name, p in out.named_parameters():
        stack = torch.stack([n[name].data.to(device=device, dtype=torch.float64) for n in named])
        p.data = (stack * w.view(-1, *([1] * (stack.dim() - 1)))).sum(0).to(p.dtype)
    return out


def capacity_ratio(profiles: list[RankProfile]) -> float:
    """``sum_k r_k / d``: the x-axis of the headline plot."""
    if not profiles:
        return float("nan")
    return sum(p.rank for p in profiles) / profiles[0].d_model


@torch.no_grad()
def fuse(
    models: list[Transformer],
    calib_tokens: list[Tensor],
    cfg: FusionConfig | None = None,
    canonicalized: bool = False,
    verbose: bool = False,
) -> tuple[Transformer, CapacityReport]:
    """Capacity-aware fusion of N aligned models.

    ``calib_tokens[k]`` must be **model k's own task data** — the effective rank
    being measured is how many directions that model needs for its own job.

    ``canonicalized=True`` asserts the caller has already run Phase 1; otherwise
    it is run here.  Returns the merged model and the report.  Repair (Phase 3)
    is deliberately *not* applied: pre- and post-repair numbers are always
    reported separately, so the caller applies it.
    """
    cfg = cfg or FusionConfig()
    require_same_arch(*models)
    if len(calib_tokens) != len(models):
        raise ValueError("need one calibration batch per model")

    if not canonicalized:
        from ..align import canonicalize_all

        models, _ = canonicalize_all(models, cfg, calib_tokens=calib_tokens[0])
    else:
        models = [m.clone() for m in models]

    profiles = [
        used_subspace(m, t.to(m.embed.device), cfg.rank_threshold, cfg.rank_measure)
        for m, t in zip(models, calib_tokens, strict=True)
    ]
    weighted = cfg.rank_measure == "full"
    report = CapacityReport(
        profiles=profiles,
        total_rank=sum(p.rank for p in profiles),
        d_model=models[0].cfg.d_model,
        participation=[participation_ratio(p.spectra[-1]) for p in profiles],
        rank_energy=[p.energy_captured for p in profiles],
        weighted=weighted,
    )
    bases = [p.basis() for p in profiles]
    # Under measure="full" the spectrum rides along with the basis and no cutoff
    # is applied anywhere; that is the whole point of the weighted objective.
    energies = [p.spectrum() for p in profiles] if weighted else None

    report.overlap_before = subspace_overlap(bases, energies)
    if weighted:
        report.interference_before = interference_ratio(bases, energies)
        report.floor = rearrangement_floor(energies)
        report.feasible = report.interference_before < KNEE
    else:
        report.feasible = report.total_rank <= report.d_model

    if cfg.disentangle and len(models) > 1:
        res: DisentangleResult = disentangle(
            bases, energies, steps=cfg.stiefel_steps, lr=cfg.stiefel_lr, verbose=verbose
        )
        for m, R in zip(models, res.rotations, strict=True):
            apply_residual_rotation(m, R.to(m.embed), check=False)
        report.overlap_after = res.overlap_after
        report.disentangled = True
        report.rotations = list(res.rotations)
        bases = [R.to(b) @ b for R, b in zip(res.rotations, bases, strict=True)]
        if weighted:
            report.interference_after = res.interference_after
            report.at_floor = res.at_floor
            report.feasible = res.feasible
    else:
        report.overlap_after = report.overlap_before
        report.interference_after = report.interference_before

    report.aligned = models
    if weighted:
        # No rank budget exists any more, so there is nothing for the budget QP
        # to select: every direction is already carried, weighted by its energy.
        # Whether the sum survives is what the interference ratio reports.
        merged = _sum_models(models)
        report.combine = "sum"
    elif report.feasible:
        # disjoint subspaces: the correct combination is the sum
        merged = _sum_models(models)
        report.combine = "sum"
    else:
        budget_basis, rho = budget_qp(
            bases, [p.spectra[-1] for p in profiles], budget=report.d_model
        )
        report.captured_energy = rho
        merged = _sum_models(models)
        _project_writers_(merged, budget_basis)
        report.combine = "sum+budget-qp"

    return merged, report


WRITERS = ("embed", "pos", "attn.w_o", "mlp.w_out")
#: readers, in the order of :func:`fusion.capacity.rank.read_point_states`
_READERS_PER_BLOCK = (("attn.w_q", "attn.w_k", "attn.w_v"), ("mlp.w_in",))


def _is_writer(name: str) -> bool:
    return any(name.endswith(w) or f".{w}" in name for w in WRITERS)


@torch.no_grad()
def _sum_models(models: list[Transformer], reader_rule: str = "mean") -> Transformer:
    """Sum the stream-facing weights; combine the readers by ``reader_rule``.

    Writers are always summed -- each writes into its own directions.  For the
    readers neither fixed rule is correct, and the two degenerate cases show why:

    ===================  ==================  ==================
    merge                readers averaged    readers summed
    ===================  ==================  ==================
    ``m`` with a *zero*  0.43  (broken)      1.00 (correct)
    ``m`` with a *copy*  1.00  (correct)     0.71 (broken)
    ===================  ==================  ==================

    Averaging scales every reader by ``1/N`` while the stream keeps its full
    magnitude, so attention scores shrink by ``N^2`` and the softmax flattens.
    Summing has the mirror problem when the models genuinely overlap.  The
    correct weight depends on how much of the merged stream is actually model
    ``k``'s signal, which is what :func:`_wiener_readers` estimates.  These
    fixed rules are kept for ablation.
    """
    out = models[0].clone()
    device = next(out.parameters()).device
    named = [dict(m.named_parameters()) for m in models]
    for name, p in out.named_parameters():
        stack = torch.stack([n[name].data.to(device=device, dtype=torch.float64) for n in named])
        take_sum = _is_writer(name) or reader_rule == "sum"
        p.data = (stack.sum(0) if take_sum else stack.mean(0)).to(p.dtype)
    return out


@torch.no_grad()
def _wiener_readers(
    models: list[Transformer],
    calib_tokens: list[Tensor],
    eps: float = 1e-10,
) -> Transformer:
    """Sum the writers; give each reader the signal it was trained to see.

    The merged stream carries ``h = sum_j h_j``.  Model ``k``'s reader wants
    ``h_k``, and the minimum-mean-squared-error linear estimate of it is the
    Wiener filter

        h_k_hat = h (sum_j C_j + eps I)^-1 C_k

    so the merged reader is ``W = sum_k (sum_j C_j)^-1 C_k W^(k)``.  Applied to
    the merged stream, each model's own reader sees its own signal and nothing
    else, to the extent the covariances are separable -- which is exactly what
    Phase 2's disentangling is trying to arrange.

    This subsumes both fixed rules rather than splitting the difference:
    against a zero model ``P_k -> I`` and it becomes a sum; against an identical
    copy ``P_k -> I/N`` and it becomes a mean.  No threshold or rank appears.

    Covariances are taken **per read point**, not per layer boundary: the
    attention and MLP blocks of one layer read the stream at different points,
    so one projector for both is the wrong projector for half the readers.

    ``eps`` is a relative eigenvalue tolerance for the pseudo-inverse -- a
    numerical rank cutoff, not a modeling knob.  Results are flat across many
    orders of magnitude of it; ``tests/test_combine_rule.py`` pins that, because
    a rule that only works at one ``eps`` would be a tuning artifact.
    """
    n = len(models)
    out = models[0].clone()
    device = next(out.parameters()).device

    # covs[k][i] = model k's second moment at read point i, on model k's own data
    # Uncentered second moments, NOT covariances.  The projector has to
    # reconstruct ``h`` itself, and centering would leave it blind along the
    # mean direction -- which is a real, functionally load-bearing component of
    # the residual stream, so zeroing the readers there destroys the model.
    covs = [
        read_point_covariance(m, t.to(device), center=False)
        for m, t in zip(models, calib_tokens, strict=True)
    ]
    n_points = len(covs[0])
    projectors = []
    for i in range(n_points):
        total = sum(covs[k][i] for k in range(n))
        # Pseudo-inverse, not a ridge.  The pooled second moment is genuinely
        # rank deficient -- at the first read point the stream spans only
        # V + T directions -- and on that null space *no* model has any signal,
        # so dropping it is exact rather than approximate.  A ridge instead
        # leaks a tunable amount of suppression into the live directions, which
        # would make the result an artifact of eps.
        evals, evecs = torch.linalg.eigh(total)
        keep = evals > evals.max().clamp_min(1e-300) * eps
        inv_evals = torch.where(keep, 1.0 / evals.clamp_min(1e-300), torch.zeros_like(evals))
        inv = (evecs * inv_evals[None, :]) @ evecs.T
        projectors.append([inv @ covs[k][i] for k in range(n)])

    named = [dict(m.named_parameters()) for m in models]

    def blended(name: str, point: int) -> Tensor:
        return sum(
            projectors[point][k].to(torch.float64) @ named[k][name].data.double()
            for k in range(n)
        )

    for name, p in out.named_parameters():
        if _is_writer(name):
            p.data = torch.stack([nm[name].data.double() for nm in named]).sum(0).to(p.dtype)
    for layer in range(out.cfg.n_layers):
        for offset, group in enumerate(_READERS_PER_BLOCK):
            point = 2 * layer + offset
            for suffix in group:
                name = f"blocks.{layer}.{suffix}"
                dict(out.named_parameters())[name].data = blended(name, point).to(
                    out.embed.dtype
                )
    if out.unembed is not None:
        out.unembed.data = blended("unembed", n_points - 1).to(out.embed.dtype)
    return out


@torch.no_grad()
def _project_writers_(model: Transformer, basis: Tensor) -> Transformer:
    """Project everything written into the residual stream onto ``basis``. In place.

    The over-capacity branch: the merged model cannot host every direction, so
    writes are restricted to the budgeted subspace and the discarded energy is
    what ``rho`` reports.
    """
    P = (basis.double() @ basis.double().T).to(model.embed.dtype)
    model.embed.data = model.embed.data @ P
    model.pos.data = model.pos.data @ P
    for block in model.blocks:
        block.attn.w_o.data = block.attn.w_o.data @ P
        block.mlp.w_out.data = block.mlp.w_out.data @ P
    return model
