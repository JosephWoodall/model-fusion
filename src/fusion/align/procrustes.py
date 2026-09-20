"""Residual-stream alignment over ``O(d)`` by orthogonal Procrustes on embeddings.

This is the hundred lines the whole program starts with.  Because every model
shares a vocabulary, the embedding matrices ``E_k`` are all expressed in the
*same token basis* — row ``t`` means token ``t`` in every model.  They differ
only by the arbitrary orthogonal frame each model happened to train into.  So
the frame is recoverable by solving

    min_{R_k orthogonal}  || E_k R_k - Ebar ||_F

with ``Ebar`` the consensus of the aligned embeddings, iterated to a fixed
point (generalized Procrustes analysis).

Two properties matter and are what make this preferable to pairwise alignment:

* **Order invariance.** The consensus is the mean of all aligned embeddings, so
  no model is privileged as the reference.
* **O(N), not O(N^2).** Each model is mapped independently into the consensus
  frame; adding an (N+1)-th model costs one more solve.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..model import Transformer, require_same_arch
from ..symmetry import apply_residual_rotation


def procrustes_rotation(source: Tensor, target: Tensor) -> Tensor:
    """The orthogonal ``R`` minimizing ``||source @ R - target||_F``.

    Closed form: with ``M = source.T @ target = U S V.T``, the optimum is
    ``R = U V.T``.  Note this is the *rotation of the source's frame into the
    target's*, not the other way round.
    """
    if source.shape != target.shape:
        raise ValueError(f"shape mismatch {tuple(source.shape)} vs {tuple(target.shape)}")
    M = source.T.to(torch.float64) @ target.to(torch.float64)
    U, _, Vh = torch.linalg.svd(M)
    return (U @ Vh).to(source.dtype)


def consensus_frame(
    embeddings: list[Tensor],
    iters: int = 30,
    tol: float = 1e-8,
) -> tuple[list[Tensor], Tensor]:
    """Generalized Procrustes: return per-model rotations and the consensus target.

    The consensus is re-estimated each round as the mean of the aligned
    embeddings, then renormalized to the mean Frobenius norm so it does not
    shrink toward zero as the models disagree.
    """
    if len(embeddings) < 2:
        raise ValueError("need at least two models to form a consensus")
    E = [e.detach().to(torch.float64) for e in embeddings]
    if any(e.shape != E[0].shape for e in E):
        raise ValueError("embedding matrices must all have the same shape")

    scale = torch.stack([e.norm() for e in E]).mean()
    target = E[0].clone()
    rotations = [torch.eye(E[0].shape[1], dtype=torch.float64) for _ in E]
    prev = float("inf")

    for _ in range(iters):
        rotations = [procrustes_rotation(e, target) for e in E]
        aligned = [e @ R for e, R in zip(E, rotations, strict=True)]
        new_target = torch.stack(aligned).mean(0)
        new_target = new_target * (scale / new_target.norm().clamp_min(1e-12))
        disagreement = sum((a - new_target).pow(2).sum().item() for a in aligned)
        shift = (new_target - target).norm().item()
        target = new_target
        if abs(prev - disagreement) < tol and shift < tol:
            break
        prev = disagreement

    dtype = embeddings[0].dtype
    return [R.to(dtype) for R in rotations], target.to(dtype)


def residual_frame(
    models: list[Transformer],
    iters: int = 30,
    tol: float = 1e-8,
    include_positions: bool = True,
    include_unembed: bool = True,
) -> list[Tensor]:
    """Per-model rotations ``R_k`` carrying each model into the shared residual frame.

    The anchor stacks every matrix whose *rows* live in a shared basis: token
    embeddings, positional embeddings (position ``i`` means position ``i`` in
    every model), and the transposed unembedding (one row per output token).
    All three are free extra rank, and rank is the binding constraint here --
    the anchor spans at most ``V + T`` of the ``d`` directions, and whatever it
    does not span, Procrustes cannot determine.  See
    :mod:`fusion.align.complement`.
    """
    require_same_arch(*models)
    from .complement import anchor_matrix

    mats = []
    for m in models:
        if include_positions and include_unembed:
            mats.append(anchor_matrix(m, include_unembed=True))
        elif include_positions:
            mats.append(torch.cat([m.embed.detach(), m.pos.detach()], dim=0))
        else:
            mats.append(m.embed.detach())
    rotations, _ = consensus_frame(mats, iters=iters, tol=tol)
    return rotations


def align_residual_(
    models: list[Transformer],
    iters: int = 30,
    tol: float = 1e-8,
) -> list[Tensor]:
    """Rotate every model into the consensus frame, in place. Returns the rotations."""
    rotations = residual_frame(models, iters=iters, tol=tol)
    for m, R in zip(models, rotations, strict=True):
        apply_residual_rotation(m, R)
    return rotations


def frame_disagreement(models: list[Transformer]) -> float:
    """Mean squared embedding disagreement after alignment — a scalar alignment score.

    Zero means the models' embeddings are the same up to the frame; large means
    the models genuinely learned different token geometry, which no rotation can
    fix and which is a capacity problem, not an alignment one.
    """
    E = torch.stack([m.embed.detach().to(torch.float64) for m in models])
    return (E - E.mean(0, keepdim=True)).pow(2).mean().item()
