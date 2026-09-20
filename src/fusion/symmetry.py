"""The exact, function-preserving reparameterizations of :class:`~fusion.model.Transformer`.

Three groups act on the network without changing any logit:

1. ``O(d)`` on the residual stream        -> :func:`apply_residual_rotation`
2. ``GL(d_head)`` inside each attention head -> :func:`apply_head_transform`
3. permutation x positive scaling on MLP hidden units -> :func:`apply_mlp_permutation`,
   :func:`apply_mlp_scaling`

Group (1) is the one this project rests on, and it exists only because the
network is bias-free with gainless RMSNorm.  Groups (2) and (3) are internal to
a sublayer, so they commute with (1); canonicalization therefore proceeds
residual -> heads -> MLP without interference.

Everything here mutates the model in place and returns it.  Tests in
``tests/test_symmetry.py`` assert that each transform leaves the model's output
unchanged to floating-point tolerance.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .model import HOMOGENEOUS_ACTIVATIONS, Transformer


def _check_orthogonal(R: Tensor, tol: float = 1e-4) -> None:
    d = R.shape[-1]
    err = (R.T @ R - torch.eye(d, dtype=R.dtype, device=R.device)).abs().max().item()
    if err > tol:
        raise ValueError(f"matrix is not orthogonal: max|RᵀR - I| = {err:.2e}")


def _require_gainless(model: Transformer) -> None:
    if any(n.gain is not None for n in model.norms()):
        raise ValueError(
            "model has active RMSNorm gains; orthogonal invariance does not hold. "
            "Call model.fold_gains() first."
        )


# --------------------------------------------------------------------------
# 1. residual stream: O(d)
# --------------------------------------------------------------------------


@torch.no_grad()
def apply_residual_rotation(model: Transformer, R: Tensor, check: bool = True) -> Transformer:
    """Rewrite the model in the rotated residual basis ``h -> h @ R``.

    ``RMSNorm(h @ R) == RMSNorm(h) @ R`` for orthogonal ``R`` because the RMS is
    rotation invariant, so every logit is unchanged.  Writers to the stream get
    a right-multiply by ``R``; readers get a left-multiply by ``Rᵀ``.
    """
    _require_gainless(model)
    R = R.to(dtype=model.embed.dtype, device=model.embed.device)
    if R.shape != (model.cfg.d_model, model.cfg.d_model):
        raise ValueError(f"R has shape {tuple(R.shape)}, expected square d_model")
    if check:
        _check_orthogonal(R)

    model.embed.data = model.embed.data @ R          # writer
    model.pos.data = model.pos.data @ R              # writer
    for block in model.blocks:
        for name in ("w_q", "w_k", "w_v"):           # readers
            w = getattr(block.attn, name)
            w.data = R.T @ w.data
        block.attn.w_o.data = block.attn.w_o.data @ R  # writer
        block.mlp.w_in.data = R.T @ block.mlp.w_in.data
        block.mlp.w_out.data = block.mlp.w_out.data @ R
    if model.unembed is not None:                    # reader
        model.unembed.data = R.T @ model.unembed.data
    return model


# --------------------------------------------------------------------------
# 2. attention heads: GL(d_head), plus a permutation across heads
# --------------------------------------------------------------------------


def qk_circuit(model: Transformer, layer: int, head: int) -> Tensor:
    """``W_q W_kᵀ`` for one head: ``[d, d]``. Attention scores depend on nothing else."""
    attn = model.blocks[layer].attn
    return attn.heads("q")[head] @ attn.heads("k")[head].T


def ov_circuit(model: Transformer, layer: int, head: int) -> Tensor:
    """``W_v W_o`` for one head: ``[d, d]``. The head's output depends on nothing else."""
    attn = model.blocks[layer].attn
    return attn.heads("v")[head] @ attn.heads("o")[head]


@torch.no_grad()
def apply_head_transform(
    model: Transformer,
    layer: int,
    head: int,
    A: Tensor | None = None,
    B: Tensor | None = None,
) -> Transformer:
    """Apply ``W_q <- W_q A``, ``W_k <- W_k A^{-T}``, ``W_v <- W_v B``, ``W_o <- B^{-1} W_o``.

    Both circuits are preserved for any invertible ``A``, ``B``, so the model's
    function is unchanged.
    """
    attn = model.blocks[layer].attn
    dh, d = attn.d_head, model.cfg.d_model
    sl = slice(head * dh, (head + 1) * dh)

    if A is not None:
        A = A.to(attn.w_q)
        A_invT = torch.linalg.inv(A).T
        attn.w_q.data[:, sl] = attn.w_q.data[:, sl] @ A
        attn.w_k.data[:, sl] = attn.w_k.data[:, sl] @ A_invT
    if B is not None:
        B = B.to(attn.w_v)
        B_inv = torch.linalg.inv(B)
        attn.w_v.data[:, sl] = attn.w_v.data[:, sl] @ B
        attn.w_o.data[sl, :] = B_inv @ attn.w_o.data[sl, :]
    del d
    return model


@torch.no_grad()
def apply_head_permutation(model: Transformer, layer: int, perm: Tensor) -> Transformer:
    """Reorder the heads of one layer. Heads are exchangeable, so this is exact."""
    attn = model.blocks[layer].attn
    perm = perm.to(torch.long)
    n, dh = attn.n_heads, attn.d_head
    if perm.shape != (n,) or sorted(perm.tolist()) != list(range(n)):
        raise ValueError("perm must be a permutation of range(n_heads)")
    idx = (perm[:, None] * dh + torch.arange(dh, device=perm.device)[None, :]).reshape(-1)
    for name in ("w_q", "w_k", "w_v"):
        w = getattr(attn, name)
        w.data = w.data[:, idx]
    attn.w_o.data = attn.w_o.data[idx, :]
    return model


# --------------------------------------------------------------------------
# 3. MLP hidden units: permutation x positive scaling
# --------------------------------------------------------------------------


@torch.no_grad()
def apply_mlp_permutation(model: Transformer, layer: int, perm: Tensor) -> Transformer:
    """Reorder MLP hidden units. Exact for any elementwise activation."""
    mlp = model.blocks[layer].mlp
    perm = perm.to(torch.long)
    h = mlp.w_in.shape[1]
    if perm.shape != (h,) or sorted(perm.tolist()) != list(range(h)):
        raise ValueError("perm must be a permutation of range(d_ff)")
    mlp.w_in.data = mlp.w_in.data[:, perm]
    mlp.w_out.data = mlp.w_out.data[perm, :]
    return model


@torch.no_grad()
def apply_mlp_scaling(model: Transformer, layer: int, scale: Tensor) -> Transformer:
    """``W_in <- W_in diag(s)``, ``W_out <- diag(1/s) W_out`` for ``s > 0``.

    Exact only for a positively homogeneous activation (ReLU).  Under GELU or
    SiLU this changes the function, so it is refused rather than silently
    approximated.
    """
    mlp = model.blocks[layer].mlp
    if mlp.activation not in HOMOGENEOUS_ACTIVATIONS:
        raise ValueError(
            f"MLP scaling is not function-preserving under {mlp.activation!r}; "
            f"exact only for {sorted(HOMOGENEOUS_ACTIVATIONS)}"
        )
    scale = scale.to(mlp.w_in)
    if (scale <= 0).any():
        raise ValueError("scale must be strictly positive")
    mlp.w_in.data = mlp.w_in.data * scale[None, :]
    mlp.w_out.data = mlp.w_out.data / scale[:, None]
    return model


# --------------------------------------------------------------------------
# random group elements, for tests and for sanity-checking alignment
# --------------------------------------------------------------------------


def random_orthogonal(d: int, seed: int | None = None, device=None, dtype=None) -> Tensor:
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=dtype or torch.float32)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))[None, :]  # fix the sign convention
    return Q.to(device) if device is not None else Q


def scramble(
    model: Transformer, seed: int = 0, heads: bool = True, mlp: bool = True
) -> Transformer:
    """Apply a random element of the full symmetry group, in place.

    Used by tests and by the "can you recover a known scramble?" diagnostic: a
    canonicalizer that is worth anything must undo this exactly.

    The group elements are generated in the model's own dtype.  Generating an
    orthogonal matrix in float32 and casting it up would make it orthogonal only
    to ~1e-7, and RMSNorm equivariance is exact only for an exactly orthogonal
    matrix -- which would show up as spurious drift in a float64 exactness test.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    cfg = model.cfg
    dtype = next(model.parameters()).dtype
    eye = torch.eye(cfg.d_head, dtype=dtype)

    apply_residual_rotation(model, random_orthogonal(cfg.d_model, seed=seed, dtype=dtype))
    for layer in range(cfg.n_layers):
        if heads:
            for h in range(cfg.n_heads):
                A = eye + torch.randn(
                    cfg.d_head, cfg.d_head, generator=g, dtype=dtype
                ) / cfg.d_head**0.5
                B = eye + torch.randn(
                    cfg.d_head, cfg.d_head, generator=g, dtype=dtype
                ) / cfg.d_head**0.5
                apply_head_transform(model, layer, h, A=A, B=B)
            apply_head_permutation(model, layer, torch.randperm(cfg.n_heads, generator=g))
        if mlp:
            apply_mlp_permutation(model, layer, torch.randperm(cfg.ff_dim, generator=g))
            if model.blocks[layer].mlp.activation in HOMOGENEOUS_ACTIVATIONS:
                s = torch.rand(cfg.ff_dim, generator=g, dtype=dtype) * 1.5 + 0.5
                apply_mlp_scaling(model, layer, s)
    return model
