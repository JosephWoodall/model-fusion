"""MLP hidden-unit canonicalization: permutation x positive scaling.

A hidden unit is defined by its input direction ``w_in[:, j]`` and its output
direction ``w_out[j, :]``.  Under a positively homogeneous activation the pair
``(w_in[:, j] * s, w_out[j, :] / s)`` computes the same thing for any ``s > 0``,
and the units may be listed in any order.

Canonical form fixes the scale by normalizing each input column to unit norm
(pushing all magnitude into the output row), which is reference-free and
per-model.  The remaining permutation is matched to a reference by activation
correlation — the standard, and better-behaved than weight matching when the
two models' units are not in one-to-one correspondence.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..model import HOMOGENEOUS_ACTIVATIONS, Transformer, require_same_arch
from ..symmetry import apply_mlp_permutation, apply_mlp_scaling

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:  # pragma: no cover
    raise ImportError("fusion.align.mlp needs scipy for the Hungarian solve") from exc


@torch.no_grad()
def canonicalize_mlp(
    model: Transformer, layer: int | None = None, eps: float = 1e-8
) -> Transformer:
    """Normalize hidden-unit scale, and order units by output energy. In place.

    Both steps are reference-free, which is what keeps canonicalization
    per-model and therefore order-invariant over N.
    """
    layers = range(model.cfg.n_layers) if layer is None else [layer]
    for li in layers:
        mlp = model.blocks[li].mlp
        if mlp.activation in HOMOGENEOUS_ACTIVATIONS:
            norms = mlp.w_in.data.norm(dim=0).clamp_min(eps)
            apply_mlp_scaling(model, li, 1.0 / norms)
        energy = mlp.w_out.data.norm(dim=1)
        apply_mlp_permutation(model, li, torch.argsort(energy, descending=True))
    return model


@torch.no_grad()
def mlp_activations(model: Transformer, layer: int, tokens: Tensor) -> Tensor:
    """Hidden activations of one MLP, flattened over batch and position: ``[B*T, d_ff]``."""
    model.eval()
    x = model.embed[tokens] + model.pos[: tokens.shape[1]]
    for i, block in enumerate(model.blocks):
        x = x + block.attn(block.ln_attn(x))
        h = block.ln_mlp(x)
        pre = h @ block.mlp.w_in
        if i == layer:
            return block.mlp.act(pre).reshape(-1, block.mlp.w_in.shape[1])
        x = x + block.mlp.act(pre) @ block.mlp.w_out
    raise IndexError(f"layer {layer} out of range")


def _corr(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Cross-correlation matrix between the columns of ``a`` and of ``b``."""
    a = a.double()
    b = b.double()
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(eps)
    b = b / b.norm(dim=0, keepdim=True).clamp_min(eps)
    return a.T @ b


def match_mlp_units(
    reference: Transformer,
    model: Transformer,
    layer: int,
    tokens: Tensor,
    by: str = "activation",
) -> Tensor:
    """Permutation aligning ``model``'s hidden units to ``reference``'s.

    ``by="activation"`` correlates the two models' hidden activations on shared
    inputs (needs data, works well).  ``by="weight"`` matches on the weights
    alone (data-free, the Git Re-Basin style objective), and is what you fall
    back to when no calibration data exists.
    """
    require_same_arch(reference, model)
    if by == "activation":
        cost = _corr(mlp_activations(reference, layer, tokens),
                     mlp_activations(model, layer, tokens))
    elif by == "weight":
        a, b = reference.blocks[layer].mlp, model.blocks[layer].mlp
        cost = (
            _corr(a.w_in.data, b.w_in.data) + _corr(a.w_out.data.T, b.w_out.data.T)
        )
    else:
        raise ValueError(f"unknown matching mode {by!r}")
    rows, cols = linear_sum_assignment(-cost.cpu().numpy())
    perm = torch.empty(cost.shape[0], dtype=torch.long)
    perm[torch.as_tensor(rows)] = torch.as_tensor(cols)
    return perm


@torch.no_grad()
def align_mlp_to(
    reference: Transformer, model: Transformer, tokens: Tensor | None = None
) -> list[Tensor]:
    """Permute ``model``'s hidden units to match ``reference`` at every layer. In place."""
    by = "activation" if tokens is not None else "weight"
    perms = []
    for li in range(model.cfg.n_layers):
        perm = match_mlp_units(reference, model, li, tokens, by=by)
        apply_mlp_permutation(model, li, perm)
        perms.append(perm)
    return perms
