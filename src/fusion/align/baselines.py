"""Published alignment methods, implemented for this architecture as comparison points.

These exist so the orthogonal-Procrustes canonicalization has something honest to
beat.  All of them search a strictly smaller group than ``O(d)``: they permute,
or they transport mass between units, but none of them can rotate the residual
stream.  That is the specific advantage the from-scratch architecture buys, and
these baselines are what measure it.

* :func:`git_rebasin_weight_matching` — Ainsworth et al., weight-space
  coordinate descent over permutations.
* :func:`activation_matching` — the same objective on activations rather than
  weights, usually a stronger baseline.
* :func:`ot_fusion` — Singh & Jaggi: Sinkhorn transport between units, a soft
  permutation.
* :func:`zipit_merge` — Stoica et al.: greedily merge correlated features
  *within and across* models, rather than assuming a bijection.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..model import Transformer, require_same_arch
from ..symmetry import apply_mlp_permutation
from .mlp import _corr, mlp_activations

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:  # pragma: no cover
    raise ImportError("fusion.align.baselines needs scipy") from exc


# --------------------------------------------------------------------------
# Git Re-Basin
# --------------------------------------------------------------------------


@torch.no_grad()
def git_rebasin_weight_matching(
    reference: Transformer,
    model: Transformer,
    iters: int = 20,
    seed: int = 0,
) -> list[Tensor]:
    """Weight matching over MLP hidden-unit permutations, in place on ``model``.

    Coordinate descent: repeatedly pick a layer and re-solve its permutation
    with the others held fixed.  Converges to a local optimum of
    ``sum_l <W_l^A, P W_l^B>``; Ainsworth et al. report this is usually enough.

    Only MLP units are permuted here.  The attention heads of this architecture
    are handled by the ``GL(d_head)`` machinery in :mod:`fusion.align.heads`,
    which is strictly more general than the permutation Re-Basin would find, so
    restricting the baseline to MLPs keeps the comparison about the residual
    frame rather than about head bookkeeping.
    """
    require_same_arch(reference, model)
    g = torch.Generator().manual_seed(seed)
    n_layers = model.cfg.n_layers
    perms = [torch.arange(model.blocks[i].mlp.w_in.shape[1]) for i in range(n_layers)]

    for _ in range(iters):
        changed = False
        for li in torch.randperm(n_layers, generator=g).tolist():
            a, b = reference.blocks[li].mlp, model.blocks[li].mlp
            cost = a.w_in.data.T.double() @ b.w_in.data.double()
            cost = cost + a.w_out.data.double() @ b.w_out.data.double().T
            rows, cols = linear_sum_assignment(-cost.cpu().numpy())
            perm = torch.empty(cost.shape[0], dtype=torch.long)
            perm[torch.as_tensor(rows)] = torch.as_tensor(cols)
            if not torch.equal(perm, torch.arange(perm.numel())):  # identity = nothing to do
                apply_mlp_permutation(model, li, perm)
                perms[li] = perms[li][perm]
                changed = True
        if not changed:
            break
    return perms


@torch.no_grad()
def activation_matching(
    reference: Transformer,
    model: Transformer,
    tokens: Tensor,
) -> list[Tensor]:
    """Match hidden units by activation correlation on shared inputs. In place.

    Usually beats weight matching, because two units can implement the same
    function with quite different weights once the layers below them differ.
    """
    require_same_arch(reference, model)
    perms = []
    for li in range(model.cfg.n_layers):
        cost = _corr(mlp_activations(reference, li, tokens), mlp_activations(model, li, tokens))
        rows, cols = linear_sum_assignment(-cost.cpu().numpy())
        perm = torch.empty(cost.shape[0], dtype=torch.long)
        perm[torch.as_tensor(rows)] = torch.as_tensor(cols)
        apply_mlp_permutation(model, li, perm)
        perms.append(perm)
    return perms


# --------------------------------------------------------------------------
# OT fusion (Sinkhorn)
# --------------------------------------------------------------------------


def sinkhorn(
    cost: Tensor,
    reg: float = 0.05,
    iters: int = 200,
    a: Tensor | None = None,
    b: Tensor | None = None,
) -> Tensor:
    """Entropic OT plan for ``cost``, with uniform marginals by default.

    ``reg -> 0`` recovers the hard permutation; larger ``reg`` gives a soft
    transport plan that averages units together, which is the whole point of OT
    fusion relative to Re-Basin.
    """
    n, m = cost.shape
    dev = cost.device
    a = a if a is not None else torch.full((n,), 1.0 / n, dtype=torch.float64, device=dev)
    b = b if b is not None else torch.full((m,), 1.0 / m, dtype=torch.float64, device=dev)
    a, b = a.to(dev), b.to(dev)
    K = torch.exp(-cost.double() / reg)
    u = torch.ones_like(a)
    for _ in range(iters):
        v = b / (K.T @ u).clamp_min(1e-300)
        u = a / (K @ v).clamp_min(1e-300)
    return torch.diag(u) @ K @ torch.diag(v)


@torch.no_grad()
def ot_fusion(
    models: list[Transformer],
    reg: float = 0.05,
    iters: int = 200,
) -> Transformer:
    """Fuse N models by transporting every model's units onto model 0's, then averaging.

    Returns a new model; inputs are untouched.
    """
    require_same_arch(*models)
    out = models[0].clone()
    ref = models[0]
    n = len(models)

    acc = {name: p.data.double().clone() for name, p in out.named_parameters()}
    for m in models[1:]:
        m = m.clone()
        for li in range(m.cfg.n_layers):
            a, b = ref.blocks[li].mlp, m.blocks[li].mlp
            h = a.w_in.shape[1]
            cost = torch.cdist(a.w_in.data.T.double(), b.w_in.data.T.double()) + torch.cdist(
                a.w_out.data.double(), b.w_out.data.double()
            )
            T = sinkhorn(cost, reg=reg, iters=iters) * h  # rescale: rows sum to 1
            b.w_in.data = (b.w_in.data.double() @ T.T).to(b.w_in.dtype)
            b.w_out.data = (T @ b.w_out.data.double()).to(b.w_out.dtype)
        for name, p in m.named_parameters():
            acc[name] += p.data.double()

    for name, p in out.named_parameters():
        p.data = (acc[name] / n).to(p.dtype)
    return out


# --------------------------------------------------------------------------
# ZipIt!
# --------------------------------------------------------------------------


@torch.no_grad()
def zipit_merge(
    models: list[Transformer],
    tokens: Tensor,
    budget: float = 1.0,
) -> Transformer:
    """ZipIt!-style feature merging on MLP hidden units.

    Unlike Re-Basin, pairs are chosen greedily by correlation from the
    *concatenated* feature set of all N models, so two units of the *same*
    model may be merged with each other when one model has redundant features.
    ``budget`` is the fraction of the original width to keep (1.0 = merge down
    to exactly ``d_ff``).

    This is the baseline that is conceptually closest to the capacity story in
    Phase 2 — it is also a packing argument — but it packs greedily in the
    unit basis rather than solving for an orthogonal frame.
    """
    require_same_arch(*models)
    out = models[0].clone()
    n = len(models)
    for li in range(out.cfg.n_layers):
        h = out.blocks[li].mlp.w_in.shape[1]
        keep = max(1, int(round(budget * h)))
        acts = torch.cat([mlp_activations(m, li, tokens) for m in models], dim=1)  # [B*T, n*h]
        w_in = torch.cat([m.blocks[li].mlp.w_in.data for m in models], dim=1).double()
        w_out = torch.cat([m.blocks[li].mlp.w_out.data for m in models], dim=0).double()

        corr = _corr(acts, acts)
        corr.fill_diagonal_(-torch.inf)
        groups = _greedy_groups(corr, target=keep, total=n * h)

        new_in = torch.stack([w_in[:, g].mean(dim=1) for g in groups], dim=1)
        new_out = torch.stack([w_out[g, :].sum(dim=0) for g in groups], dim=0)
        out.blocks[li].mlp.w_in.data = new_in.to(out.blocks[li].mlp.w_in.dtype)
        out.blocks[li].mlp.w_out.data = new_out.to(out.blocks[li].mlp.w_out.dtype)

    # everything outside the MLPs is averaged
    for name, p in out.named_parameters():
        if ".mlp." in name:
            continue
        p.data = torch.stack(
            [dict(m.named_parameters())[name].data.to(p.device) for m in models]
        ).mean(0).to(p.dtype)
    return out


def _greedy_groups(corr: Tensor, target: int, total: int) -> list[list[int]]:
    """Greedily merge the most correlated pair until ``target`` groups remain."""
    groups = [[i] for i in range(total)]
    alive = list(range(total))
    C = corr.clone()
    while len(alive) > target:
        sub = C[alive][:, alive]
        flat = torch.argmax(sub).item()
        i, j = divmod(flat, len(alive))
        if i == j:
            break
        gi, gj = alive[i], alive[j]
        groups[gi] = groups[gi] + groups[gj]
        # the merged group's correlation with the rest is the mean of its members'
        C[gi] = (C[gi] + C[gj]) / 2
        C[:, gi] = (C[:, gi] + C[:, gj]) / 2
        C[gi, gi] = -torch.inf
        alive.remove(gj)
    return [groups[i] for i in alive]
