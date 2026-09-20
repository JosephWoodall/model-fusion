"""Fixing the residual frame *outside* the span of the shared-vocabulary anchor.

Procrustes on the embeddings is the right idea and it is not sufficient, for a
reason that is easy to miss and easy to measure.

The anchor matrices -- token embeddings ``E``, positional embeddings ``P``, and
the unembedding ``U^T`` -- are the only things expressed in a basis shared
across models.  They span at most ``V + T`` directions.  Whenever ``d`` exceeds
that, **Procrustes determines the rotation only on a proper subspace** and
leaves it completely arbitrary on the orthogonal complement.

Those complement directions are not inert.  Layer 0 activations do live entirely
in the anchor span, by construction -- but attention and MLP outputs write
wherever they like, and in the testbed here (``d=128``, ``V+T=57``) **over half
of the residual-stream energy at layers 1 and 2 sits outside the anchor span**.
So the embedding-only alignment can drive embedding disagreement to machine zero
while the interpolation barrier does not move at all, which is exactly the
symptom that would otherwise be misread as "the models learned different
solutions".

The fix keeps canonicalization reference-free.  On the complement there is no
shared basis to match against, so instead each model is put into *its own*
canonical frame there: diagonalize the residual-stream activation covariance
restricted to the complement, order the eigenvectors by eigenvalue, and fix each
sign by the skewness of the projected activations.  Two models that differ only
by a rotation land on the same frame; two models that genuinely differ land on
frames that are at least comparably ordered.

The remaining caveat is honest and reported rather than hidden: where two
eigenvalues are nearly equal the eigenvector pair is ill-determined, and
:func:`complement_conditioning` measures how much of that there is.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..capacity.rank import residual_states
from ..model import Transformer, require_same_arch
from ..symmetry import apply_residual_rotation


def anchor_matrix(model: Transformer, include_unembed: bool = True) -> Tensor:
    """Every matrix of ``model`` whose *rows* are in a basis shared across models.

    ``E`` (one row per token), ``P`` (one row per position), and optionally
    ``U^T`` (one row per output token).  These are the only handles Procrustes
    has, and stacking all of them is free extra rank.
    """
    parts = [model.embed.detach(), model.pos.detach()]
    if include_unembed and model.unembed is not None:
        parts.append(model.unembed.detach().T)
    return torch.cat(parts, dim=0)


def anchor_basis(models: list[Transformer], tol: float = 1e-8) -> Tensor:
    """Orthonormal basis ``[d, r]`` of the span the anchor pins down.

    Computed from the *consensus* anchor, so it is one basis shared by all
    models rather than a per-model choice.  Call this only after the models have
    been brought into the consensus frame.
    """
    require_same_arch(*models)
    A = torch.stack([anchor_matrix(m).double() for m in models]).mean(0)
    U, S, _ = torch.linalg.svd(A.T, full_matrices=False)       # columns of A.T span the frame
    r = int((S > S[0] * tol).sum().item())
    return U[:, :r].to(models[0].embed.dtype)


def orthogonal_complement(basis: Tensor) -> Tensor:
    """Orthonormal basis ``[d, d-r]`` of the complement of ``basis``'s column span."""
    d, r = basis.shape
    Q, _ = torch.linalg.qr(basis.double(), mode="complete")
    return Q[:, r:].to(basis.dtype) if r < d else basis.new_zeros(d, 0)


@torch.no_grad()
def pooled_covariance(model: Transformer, tokens: Tensor, skip_first: bool = True) -> Tensor:
    """Residual covariance pooled over layers, each layer normalized to unit trace.

    Normalizing per layer stops the last layer -- whose activations are largest
    -- from deciding the frame on its own.  The layer-0 covariance is skipped by
    default: it lies entirely inside the anchor span and so contributes nothing
    to the complement.
    """
    states = residual_states(model, tokens)
    if skip_first:
        states = states[1:]
    d = model.cfg.d_model
    total = torch.zeros(d, d, dtype=torch.float64, device=tokens.device)
    for h in states:
        flat = h.reshape(-1, d).double()
        flat = flat - flat.mean(0, keepdim=True)
        cov = flat.T @ flat / max(1, flat.shape[0])
        total += cov / cov.diagonal().sum().clamp_min(1e-30)
    return total


@torch.no_grad()
def complement_frame(
    model: Transformer,
    tokens: Tensor,
    anchor: Tensor,
    complement: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """The rotation carrying ``model`` into its canonical frame on the complement.

    Returns ``(R, eigenvalues)``.  ``R`` is ``[d, d]`` orthogonal, acting as the
    identity-in-spirit on the anchor span (it re-expresses it in the shared
    ``anchor`` coordinates, identically for every model) and as the covariance
    eigenbasis on the complement.
    """
    Qc = orthogonal_complement(anchor) if complement is None else complement
    if Qc.shape[1] == 0:
        return torch.eye(anchor.shape[0], dtype=anchor.dtype, device=anchor.device), \
            anchor.new_zeros(0)

    cov = pooled_covariance(model, tokens)
    M = Qc.T.double() @ cov @ Qc.double()
    evals, evecs = torch.linalg.eigh(M)
    order = torch.argsort(evals, descending=True)
    evals, W = evals[order], evecs[:, order]

    # sign convention: make the projected activations positively skewed
    states = residual_states(model, tokens)
    flat = torch.cat([h.reshape(-1, h.shape[-1]) for h in states[1:]], dim=0).double()
    flat = flat - flat.mean(0, keepdim=True)
    proj = flat @ (Qc.double() @ W)
    skew = proj.pow(3).mean(0)
    fallback = torch.sign(W[W.abs().argmax(0), torch.arange(W.shape[1], device=W.device)])
    signs = torch.where(skew.abs() > 1e-12, torch.sign(skew), fallback)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    W = W * signs[None, :]

    R = torch.cat([anchor.double(), Qc.double() @ W], dim=1)
    return R.to(anchor.dtype), evals.to(anchor.dtype)


@torch.no_grad()
def complete_residual_frame_(
    models: list[Transformer],
    calib_tokens: Tensor | list[Tensor],
    tol: float = 1e-8,
) -> tuple[list[Tensor], Tensor]:
    """Pin the residual frame on the complement of the anchor span, in place.

    Run this *after* :func:`fusion.align.procrustes.align_residual_`.  Returns
    the applied rotations and the shared anchor basis.
    """
    require_same_arch(*models)
    if isinstance(calib_tokens, Tensor):
        calib_tokens = [calib_tokens] * len(models)
    if len(calib_tokens) != len(models):
        raise ValueError("need one calibration batch per model, or one shared batch")

    basis = anchor_basis(models, tol=tol)
    Qc = orthogonal_complement(basis)
    rotations = []
    for m, toks in zip(models, calib_tokens, strict=True):
        R, _ = complement_frame(m, toks.to(m.embed.device), basis.to(m.embed), Qc.to(m.embed))
        apply_residual_rotation(m, R, check=False)
        rotations.append(R)
    return rotations, basis


@torch.no_grad()
def complement_conditioning(
    model: Transformer,
    tokens: Tensor,
    anchor: Tensor,
) -> dict[str, float]:
    """How well determined the complement frame is. Report it; do not hide it.

    ``min_relative_gap`` near zero means two covariance eigenvalues are nearly
    equal and the corresponding directions are interchangeable, so any
    conclusion resting on that pair is fragile.  ``energy_fraction`` is the share
    of residual energy living outside the anchor span at all -- if it is small,
    none of this matters and plain Procrustes was enough.
    """
    Qc = orthogonal_complement(anchor)
    cov = pooled_covariance(model, tokens)
    inside = torch.diagonal(anchor.T.double() @ cov @ anchor.double()).sum().item()
    if Qc.shape[1] == 0:
        return {"energy_fraction": 0.0, "min_relative_gap": 1.0, "complement_dim": 0}
    M = Qc.T.double() @ cov @ Qc.double()
    outside = torch.diagonal(M).sum().item()
    evals = torch.linalg.eigvalsh(M).flip(0).clamp_min(0)
    live = evals[evals > evals[0] * 1e-6]
    gaps = ((live[:-1] - live[1:]) / live[:-1].clamp_min(1e-30)) if live.numel() > 1 \
        else torch.ones(1, dtype=torch.float64, device=evals.device)
    return {
        "energy_fraction": outside / max(inside + outside, 1e-30),
        "min_relative_gap": float(gaps.min()),
        "complement_dim": int(Qc.shape[1]),
    }
