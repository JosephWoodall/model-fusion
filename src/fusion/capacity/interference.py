"""Measuring the interference that actually reaches a merged model.

:mod:`fusion.capacity.stiefel` minimizes overlap between the covariances
``C_k``, each measured on **model k's own task data**.  That is the quantity the
capacity story is written in terms of, and it is not the quantity that hurts.

Inside the merged model, every model's machinery runs on *whatever input
arrives*.  When task ``k`` is presented, model ``j``'s attention and MLPs do not
sit idle -- they respond to task ``k``'s tokens and write their response into
the shared residual stream.  The energy that collides with model ``k``'s signal
is therefore governed by

    C_j^(k) = Cov[ model j's residual stream, on task k's data ]

not by ``C_j``.  Those two can differ enormously: a model fed out-of-
distribution input is not quiet, it is *loud and wrong*.

This module measures both, so the gap between them is visible:

``self_interference``
    What the optimizer minimizes: ``sum_{j != k} tr(A_k A_j) / tr(A_k^2)``,
    every covariance on its own data.

``cross_interference``
    What the merged model experiences: ``sum_{j != k} tr(A_k A_j^(k)) /
    tr(A_k^2)``, with model ``j``'s covariance measured on task ``k``'s data.

If the second is large while the first is near zero, the disentangling
objective is optimizing a proxy that does not bind, and no amount of solver
effort will fix the merge.  Costs ``N^2`` forward passes, which is why it is a
diagnostic rather than part of the objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..model import Transformer, require_same_arch
from .rank import activation_covariance


@dataclass
class InterferenceReport:
    """Self- versus cross-interference, per model and worst-case."""

    self_per_model: list[float]
    cross_per_model: list[float]

    @property
    def self_worst(self) -> float:
        return max(self.self_per_model) if self.self_per_model else float("nan")

    @property
    def cross_worst(self) -> float:
        return max(self.cross_per_model) if self.cross_per_model else float("nan")

    @property
    def gap(self) -> float:
        """How far the proxy understates the real thing. 1.0 means it is faithful."""
        s = self.self_worst
        return self.cross_worst / s if s > 1e-12 else float("inf")

    def __str__(self) -> str:
        return (
            f"interference  self {self.self_worst:.3f}  cross {self.cross_worst:.3f}  "
            f"(cross/self = {self.gap:.1f}x)"
        )


@torch.no_grad()
def _pooled_cov(model: Transformer, tokens: Tensor) -> Tensor:
    """Trace-normalized pooled residual covariance, matching ``RankProfile``'s convention."""
    covs = activation_covariance(model, tokens)
    out = torch.zeros_like(covs[0])
    for cov in covs:
        out = out + cov / cov.diagonal().sum().clamp_min(1e-30)
    return out / len(covs)


@torch.no_grad()
def measure_interference(
    models: list[Transformer],
    calib_tokens: list[Tensor],
) -> InterferenceReport:
    """Compare the interference the objective minimizes against the one that hurts.

    ``models`` must already be in the merged frame -- that is, after
    canonicalization *and* after the disentangling rotations have been applied,
    which is the state :func:`fusion.capacity.fuse` leaves them in.  Measuring
    before that would compare the wrong frames.

    ``calib_tokens[k]`` is model ``k``'s own task data.
    """
    require_same_arch(*models)
    n = len(models)
    if len(calib_tokens) != n:
        raise ValueError("need one calibration batch per model")
    device = next(models[0].parameters()).device
    toks = [t.to(device) for t in calib_tokens]

    # C[j][k] = covariance of model j's residual stream on task k's data.
    # The diagonal C[k][k] is what the objective uses; the column is what model
    # k actually collides with inside the merged model.
    C = [[_pooled_cov(models[j], toks[k]) for k in range(n)] for j in range(n)]

    self_ratios, cross_ratios = [], []
    for k in range(n):
        Ak = C[k][k]
        denom = (Ak * Ak).sum().clamp_min(1e-30)
        self_ratios.append(
            float(sum((Ak * C[j][j]).sum() for j in range(n) if j != k) / denom)
        )
        cross_ratios.append(
            float(sum((Ak * C[j][k]).sum() for j in range(n) if j != k) / denom)
        )
    return InterferenceReport(self_per_model=self_ratios, cross_per_model=cross_ratios)


@torch.no_grad()
def signal_retention(
    merged: Transformer,
    specialist: Transformer,
    tokens: Tensor,
) -> list[float]:
    """Per-layer cosine similarity between the merged and specialist residual streams.

    The most direct possible statement of whether a merge preserved a model:
    run task ``k``'s data through both and ask whether the merged model's
    residual still points where the specialist's did.  1.0 means the signal
    survived the sum; near 0 means it was buried, whatever the covariance
    geometry said.
    """
    from .rank import residual_states

    device = next(merged.parameters()).device
    tokens = tokens.to(device)
    hm = residual_states(merged, tokens)
    hs = residual_states(specialist.to(device), tokens)
    out = []
    for a, b in zip(hm, hs, strict=True):
        x = a.reshape(-1, a.shape[-1]).double()
        y = b.reshape(-1, b.shape[-1]).double()
        cos = (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-30)
        out.append(float(cos.mean()))
    return out
