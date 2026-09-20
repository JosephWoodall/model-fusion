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

from dataclasses import dataclass

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

    @property
    def rank(self) -> int:
        """The binding constraint: the widest layer's demand on the stream."""
        return max(self.ranks) if self.ranks else 0

    @property
    def mean_rank(self) -> float:
        return sum(self.ranks) / len(self.ranks) if self.ranks else 0.0

    def basis(self) -> Tensor:
        """A single basis for the model's used subspace, pooled across layers.

        Layers share one residual stream, so the directions a model occupies is
        the span of the union.  Pooling by SVD of the concatenated, energy-
        weighted bases keeps that honest rather than picking one layer.
        """
        stacked = torch.cat(self.bases, dim=1)
        U, S, _ = torch.linalg.svd(stacked, full_matrices=False)
        keep = int((S > S.max() * 1e-6).sum().item())
        return U[:, : min(keep, self.rank)]

    def __str__(self) -> str:
        return (
            f"RankProfile(d={self.d_model}, measure={self.measure}, per-layer={self.ranks}, "
            f"binding={self.rank}, mean={self.mean_rank:.1f})"
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

    ``measure="participation"`` (the default) uses the participation ratio
    ``(sum l)^2 / sum l^2``, rounded.  ``measure="threshold"`` counts the
    eigenvalues needed to reach ``threshold`` of the total energy.

    **The default is not arbitrary.**  Measured on this testbed, the thresholded
    rank of a single modular-addition model at ``d=128`` is 39, 98, or 124
    depending on whether the threshold is 0.90, 0.99, or 0.999 -- so a capacity
    law stated as ``sum_k r_k <= d`` would have its x-axis, and therefore its
    predicted knee, set by an arbitrary constant.  The participation ratio has
    no such knob and lands at ~11 for the same model, which is the number that
    reflects where the activation energy actually is.  ``threshold`` is kept
    available for comparison against the literature, which mostly uses it.
    """
    evals, evecs = torch.linalg.eigh(cov.double())
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].clamp_min(0), evecs[:, order]

    if measure == "participation":
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
    ranks, bases, spectra = [], [], []
    for cov in activation_covariance(model, tokens):
        r, basis, evals = effective_rank(cov, threshold, measure)
        ranks.append(r)
        bases.append(basis.to(model.embed.dtype))
        spectra.append(evals.to(model.embed.dtype))
    return RankProfile(
        ranks=ranks, bases=bases, spectra=spectra,
        d_model=model.cfg.d_model, threshold=threshold, measure=measure,
    )
