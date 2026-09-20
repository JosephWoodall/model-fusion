"""Effective rank of the residual stream: how much of ``d`` a model actually uses.

The premise of the whole capacity argument is that a small model trained on a
narrow task uses far fewer than ``d`` directions.  That slack is what makes
packing N models into one width-``d`` residual stream conceivable.  This module
measures the slack.

For model ``k`` at a given layer, run *that model's own task data*, take the
covariance of the residual stream, and count the eigenvalues needed to reach a
target fraction of the total energy.  Using each model's own data is essential:
the question is how many directions that model needs *for its job*, not how it
responds to someone else's inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..model import Transformer


@dataclass
class RankProfile:
    """Per-layer effective rank and the subspace basis that realizes it."""

    ranks: list[int]
    bases: list[Tensor]          # each [d, r_l], orthonormal columns
    spectra: list[Tensor]        # eigenvalues, descending
    d_model: int
    threshold: float
    measure: str = "participation"
    captured: list[float] = field(default_factory=list)   # energy fraction inside each basis
    pooled_basis: Tensor | None = None    # [d, d] eigenvectors of the pooled covariance
    pooled_spectrum: Tensor | None = None  # [d] eigenvalues, descending

    @property
    def rank(self) -> int:
        """The binding constraint: the widest layer's demand on the stream."""
        return max(self.ranks) if self.ranks else 0

    @property
    def mean_rank(self) -> float:
        return sum(self.ranks) / len(self.ranks) if self.ranks else 0.0

    @property
    def energy_captured(self) -> float:
        """Fraction of activation energy inside the binding layer's basis.

        **Read this before believing any rank.** A rank that captures 75% of the
        energy is not a description of the subspace the model uses -- the other
        25% is spread over directions that still carry the computation, and
        making the *reported* subspaces disjoint leaves those colliding. That
        is exactly the failure mode measured in ``docs/FINDINGS.md``: overlap
        driven to 0.000 on participation-ratio bases while the 99%-energy bases
        still overlapped at 0.77.
        """
        if not self.captured:
            return float("nan")
        return self.captured[self.ranks.index(self.rank)]

    def warn_if_low_energy(self, floor: float = 0.95) -> str | None:
        """A one-line caveat when the rank does not account for the activations."""
        e = self.energy_captured
        if e != e or e >= floor:
            return None
        return (
            f"rank {self.rank} captures only {100 * e:.1f}% of activation energy; "
            f"sum(r_k) <= d is not a meaningful capacity test at this coverage"
        )

    def basis(self) -> Tensor:
        """A single basis for the model's used subspace, pooled across layers.

        Layers share one residual stream, so the directions a model occupies is
        the span of the union.  Pooling by SVD of the concatenated, energy-
        weighted bases keeps that honest rather than picking one layer.

        Under ``measure="full"`` the pooled covariance eigenbasis is returned
        directly, so that :meth:`spectrum` lines up column for column with it --
        which the energy-weighted objective requires.
        """
        if self.measure == "full" and self.pooled_basis is not None:
            return self.pooled_basis
        stacked = torch.cat(self.bases, dim=1)
        U, S, _ = torch.linalg.svd(stacked, full_matrices=False)
        keep = int((S > S.max() * 1e-6).sum().item())
        return U[:, : min(keep, self.rank)]

    def spectrum(self) -> Tensor:
        """Eigenvalues matching :meth:`basis` column for column.

        This is what makes the objective energy-weighted rather than
        rank-thresholded: each direction is carried with the variance actually
        on it.  Only meaningful under ``measure="full"``, where basis and
        spectrum come from the same eigendecomposition; the cutoff measures
        pool bases across layers by SVD, which severs that correspondence.
        """
        if self.pooled_spectrum is None:
            raise RuntimeError(
                "no pooled spectrum on this profile; build it with measure='full'"
            )
        return self.pooled_spectrum[: self.basis().shape[1]]

    def __str__(self) -> str:
        return (
            f"RankProfile(d={self.d_model}, measure={self.measure}, per-layer={self.ranks}, "
            f"binding={self.rank}, mean={self.mean_rank:.1f}, "
            f"energy={100 * self.energy_captured:.1f}%)"
        )


@torch.no_grad()
def residual_states(model: Transformer, tokens: Tensor) -> list[Tensor]:
    """Residual stream at every layer boundary: ``n_layers+1`` tensors of ``[B, T, d]``."""
    model.eval()
    x = model.embed[tokens] + model.pos[: tokens.shape[1]]
    out = [x]
    for block in model.blocks:
        x = block(x)
        out.append(x)
    return out


@torch.no_grad()
def read_point_states(model: Transformer, tokens: Tensor) -> list[Tensor]:
    """Residual stream at every point where a matrix *reads* from it.

    Order is ``[block0.ln_attn, block0.ln_mlp, block1.ln_attn, ..., ln_f]`` --
    ``2 * n_layers + 1`` tensors of ``[B, T, d]``, matching
    :meth:`fusion.model.Transformer.norms`.

    Layer boundaries are not the right places to measure for a merge: the
    attention block and the MLP block of the same layer read the stream at
    *different* points, because the MLP sees the attention output added in.  A
    projector built from layer-boundary statistics is therefore the wrong
    projector for half the readers.
    """
    model.eval()
    x = model.embed[tokens] + model.pos[: tokens.shape[1]]
    out = []
    for block in model.blocks:
        out.append(x)                       # ln_attn reads here
        x = x + block.attn(block.ln_attn(x))
        out.append(x)                       # ln_mlp reads here
        x = x + block.mlp(block.ln_mlp(x))
    out.append(x)                           # ln_f reads here
    return out


@torch.no_grad()
def read_point_covariance(model: Transformer, tokens: Tensor, center: bool = True) -> list[Tensor]:
    """Covariance at each read point, in the order of :func:`read_point_states`."""
    covs = []
    for h in read_point_states(model, tokens):
        flat = h.reshape(-1, h.shape[-1]).double()
        if center:
            flat = flat - flat.mean(0, keepdim=True)
        covs.append(flat.T @ flat / max(1, flat.shape[0]))
    return covs


@torch.no_grad()
def activation_covariance(model: Transformer, tokens: Tensor, center: bool = True) -> list[Tensor]:
    """Residual-stream covariance at each layer: ``n_layers+1`` matrices of ``[d, d]``.

    Centering is on by default.  The uncentered second moment would count the
    mean direction as a used direction, which inflates every rank by one and
    would make the ``sum r_k <= d`` threshold systematically pessimistic.
    """
    covs = []
    for h in residual_states(model, tokens):
        flat = h.reshape(-1, h.shape[-1]).double()
        if center:
            flat = flat - flat.mean(0, keepdim=True)
        covs.append(flat.T @ flat / max(1, flat.shape[0]))
    return covs


def effective_rank(
    cov: Tensor,
    threshold: float = 0.99,
    measure: str = "participation",
) -> tuple[int, Tensor, Tensor]:
    """Number of directions the activations actually occupy.

    Returns ``(r, basis[d, r], eigenvalues)``.

    ``measure="full"`` keeps every direction and applies no cutoff at all; the
    spectrum is then carried into the energy-weighted overlap objective, which
    is how Phase 2 avoids having a cutoff decide its answer.  This is the
    default for fusion.

    ``measure="participation"`` uses the participation ratio
    ``(sum l)^2 / sum l^2``, rounded.  ``measure="threshold"`` counts the
    eigenvalues needed to reach ``threshold`` of the total energy.

    **Why "full" is the default.**  Both cutoffs turned out to be untestable
    (``docs/FINDINGS.md``).  The thresholded rank of a single modular-addition
    model at ``d=128`` is 39, 98, or 124 depending on whether the threshold is
    0.90, 0.99, or 0.999, so ``sum_k r_k <= d`` has its predicted knee set by an
    arbitrary constant.  The participation ratio has no knob but is far too
    generous: its basis captured only 74-77% of activation energy, and
    disentangling it to *exactly* zero overlap left the 99%-energy bases still
    colliding at 0.77.  Carrying the whole spectrum and weighting by energy
    removes the choice entirely.  Both cutoffs are kept for comparison, and
    ``threshold`` is what most of the literature reports.
    """
    evals, evecs = torch.linalg.eigh(cov.double())
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].clamp_min(0), evecs[:, order]

    if measure == "full":
        r = evals.numel()
    elif measure == "participation":
        r = max(1, int(round(participation_ratio(evals))))
    elif measure == "threshold":
        if threshold >= 1.0:
            r = int((evals > evals[0] * 1e-6).sum().item())
        else:
            total = evals.sum().clamp_min(1e-30)
            r = int(torch.searchsorted(torch.cumsum(evals, 0) / total, threshold).item()) + 1
    else:
        raise ValueError(f"unknown rank measure {measure!r}")
    r = min(max(r, 1), evals.numel())
    return r, evecs[:, :r], evals


def participation_ratio(evals: Tensor) -> float:
    """``(sum l)^2 / sum l^2`` — a threshold-free effective dimension.

    Reported alongside the thresholded rank so that a conclusion never rests on
    one arbitrary choice of ``threshold``.
    """
    e = evals.clamp_min(0).double()
    return float(e.sum().pow(2) / e.pow(2).sum().clamp_min(1e-30))


@torch.no_grad()
def used_subspace(
    model: Transformer,
    tokens: Tensor,
    threshold: float = 0.99,
    measure: str = "participation",
) -> RankProfile:
    """Full rank profile of a model on its own data."""
    covs = activation_covariance(model, tokens)
    ranks, bases, spectra, captured = [], [], [], []
    for cov in covs:
        r, basis, evals = effective_rank(cov, threshold, measure)
        ranks.append(r)
        bases.append(basis.to(model.embed.dtype))
        spectra.append(evals.to(model.embed.dtype))
        e = evals.clamp_min(0)
        captured.append(float(e[:r].sum() / e.sum().clamp_min(1e-30)))

    # Pooled covariance, each layer trace-normalized so the last layer -- whose
    # activations are largest -- does not decide the frame by itself.
    pooled = torch.zeros_like(covs[0])
    for cov in covs:
        pooled = pooled + cov / cov.diagonal().sum().clamp_min(1e-30)
    pooled = pooled / len(covs)
    pe, pv = torch.linalg.eigh(pooled)
    order = torch.argsort(pe, descending=True)

    return RankProfile(
        ranks=ranks, bases=bases, spectra=spectra, captured=captured,
        d_model=model.cfg.d_model, threshold=threshold, measure=measure,
        pooled_basis=pv[:, order].to(model.embed.dtype),
        pooled_spectrum=pe[order].clamp_min(0).to(model.embed.dtype),
    )
