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
from .rank import RankProfile, participation_ratio, used_subspace
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
        bases = [R.to(b) @ b for R, b in zip(res.rotations, bases, strict=True)]
        if weighted:
            report.interference_after = res.interference_after
            report.at_floor = res.at_floor
            report.feasible = res.feasible
    else:
        report.overlap_after = report.overlap_before
        report.interference_after = report.interference_before

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


@torch.no_grad()
def _sum_models(models: list[Transformer]) -> Transformer:
    """Sum the stream-facing weights, average the rest.

    Readers (``W_q``, ``W_k``, ``W_v``, ``W_in``, ``U``) are averaged, not
    summed: a reader restricted to model ``k``'s subspace already ignores the
    other models' directions, so averaging preserves its response while summing
    would scale it by N.  Writers are summed, because each writes into its own
    orthogonal set of directions.
    """
    out = models[0].clone()
    device = next(out.parameters()).device
    named = [dict(m.named_parameters()) for m in models]
    writers = ("embed", "pos", "attn.w_o", "mlp.w_out")
    for name, p in out.named_parameters():
        stack = torch.stack([n[name].data.to(device=device, dtype=torch.float64) for n in named])
        is_writer = any(name.endswith(w) or f".{w}" in name for w in writers)
        p.data = (stack.sum(0) if is_writer else stack.mean(0)).to(p.dtype)
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
