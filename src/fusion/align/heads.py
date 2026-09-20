"""Attention-head canonicalization over ``GL(d_head)``, then a permutation across heads.

Attention depends on ``W_q, W_k`` only through the QK circuit ``C_qk = W_q W_k^T``
and on ``W_v, W_o`` only through the OV circuit ``C_ov = W_v W_o``.  Both are
``[d, d]`` matrices of rank at most ``d_head``.  So the head's *content* is the
pair of circuits, and the factorization into ``(W_q, W_k)`` is pure gauge.

The canonical factorization is the one the SVD gives::

    C = U S V^T  (truncated to rank d_head)  ->  W_q := U_r S_r^{1/2},  W_k := V_r S_r^{1/2}

Two heads computing the same circuit land on the same representative whatever
basis they trained in, so after canonicalization heads differ across models only
by *which head is which* — a permutation, solved with Hungarian on circuit
similarity.

Sign/rotation degeneracy: the SVD is unique only up to the sign of each singular
vector pair (and up to an arbitrary rotation within a degenerate singular
subspace).  ``_fix_signs`` pins the signs deterministically.  Exactly degenerate
singular values are rare in trained weights; where they occur the residual
ambiguity is reported by :func:`canonicalization_residual` rather than hidden.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..model import Transformer, require_same_arch
from ..symmetry import apply_head_permutation, ov_circuit, qk_circuit

try:  # scipy is a hard dep, but keep the import local to a helpful message
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:  # pragma: no cover
    raise ImportError("fusion.align.heads needs scipy for the Hungarian solve") from exc


def _fix_signs(U: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
    """Pin the sign of each singular-vector pair: make the largest-|.| entry of U positive."""
    idx = U.abs().argmax(dim=0)
    signs = torch.sign(U[idx, torch.arange(U.shape[1], device=U.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return U * signs[None, :], V * signs[None, :]


def canonical_factors(circuit: Tensor, rank: int) -> tuple[Tensor, Tensor, Tensor]:
    """Factor ``C ~= L @ R^T`` with ``L, R`` in canonical (singular-vector) coordinates.

    Returns ``(L, R, singular_values)`` where ``L = U_r S_r^{1/2}`` and
    ``R = V_r S_r^{1/2}``.
    """
    U, S, Vh = torch.linalg.svd(circuit.to(torch.float64), full_matrices=False)
    U, S, V = U[:, :rank], S[:rank], Vh[:rank].T
    U, V = _fix_signs(U, V)
    root = S.clamp_min(0).sqrt()
    return (U * root[None, :]).to(circuit.dtype), (V * root[None, :]).to(circuit.dtype), S.to(
        circuit.dtype
    )


@torch.no_grad()
def canonicalize_heads(model: Transformer, layer: int | None = None) -> Transformer:
    """Rewrite every head of the model in its own circuit-singular basis, in place.

    Exact: both circuits are reconstructed to SVD precision, so the model's
    function is unchanged.
    """
    layers = range(model.cfg.n_layers) if layer is None else [layer]
    dh = model.cfg.d_head
    for li in layers:
        attn = model.blocks[li].attn
        new_q, new_k, new_v, new_o = [], [], [], []
        for h in range(attn.n_heads):
            q, k, _ = canonical_factors(qk_circuit(model, li, h), dh)
            v, o, _ = canonical_factors(ov_circuit(model, li, h), dh)
            new_q.append(q)
            new_k.append(k)
            new_v.append(v)
            new_o.append(o.T)  # OV writes out: [d_head, d]
        attn.w_q.data = torch.cat(new_q, dim=1)
        attn.w_k.data = torch.cat(new_k, dim=1)
        attn.w_v.data = torch.cat(new_v, dim=1)
        attn.w_o.data = torch.cat(new_o, dim=0)
    return model


def head_signature(model: Transformer, layer: int, head: int) -> Tensor:
    """A basis-free descriptor of a head: the concatenated circuit spectra.

    Singular values are invariant to the ``GL(d_head)`` gauge *and* to the
    residual rotation (which conjugates the circuit by an orthogonal matrix), so
    this is the right thing to match heads on.
    """
    _, _, s_qk = canonical_factors(qk_circuit(model, layer, head), model.cfg.d_head)
    _, _, s_ov = canonical_factors(ov_circuit(model, layer, head), model.cfg.d_head)
    return torch.cat([s_qk, s_ov])


def head_similarity(a: Transformer, b: Transformer, layer: int) -> Tensor:
    """``[n_heads, n_heads]`` similarity between the heads of two models at one layer.

    Uses the full circuits rather than only the spectra, because by this point
    both models are already in the shared residual frame, so the circuits are
    directly comparable entry by entry.
    """
    require_same_arch(a, b)
    n = a.cfg.n_heads
    sim = torch.zeros(n, n, dtype=torch.float64, device=a.embed.device)
    ca = [(qk_circuit(a, layer, i).double(), ov_circuit(a, layer, i).double()) for i in range(n)]
    cb = [(qk_circuit(b, layer, j).double(), ov_circuit(b, layer, j).double()) for j in range(n)]
    for i, (qa, oa) in enumerate(ca):
        for j, (qb, ob) in enumerate(cb):
            num = (qa * qb).sum() + (oa * ob).sum()
            den = (qa.norm() * qb.norm() + oa.norm() * ob.norm()).clamp_min(1e-12)
            sim[i, j] = num / den
    return sim


def match_heads(a: Transformer, b: Transformer, layer: int) -> Tensor:
    """Permutation aligning ``b``'s heads to ``a``'s, by Hungarian on circuit similarity."""
    sim = head_similarity(a, b, layer)
    rows, cols = linear_sum_assignment(-sim.cpu().numpy())
    perm = torch.empty(a.cfg.n_heads, dtype=torch.long)
    perm[torch.as_tensor(rows)] = torch.as_tensor(cols)
    return perm


@torch.no_grad()
def align_heads_to(reference: Transformer, model: Transformer) -> list[Tensor]:
    """Permute ``model``'s heads, layer by layer, to match ``reference``. In place."""
    perms = []
    for li in range(model.cfg.n_layers):
        perm = match_heads(reference, model, li)
        apply_head_permutation(model, li, perm)
        perms.append(perm)
    return perms


def canonical_head_order(model: Transformer, layer: int) -> Tensor:
    """A reference-free head ordering: sort by circuit energy, descending.

    Used when no reference model is available, so that canonicalization stays
    strictly per-model.  It is weaker than Hungarian matching — it only breaks
    the permutation ambiguity when the heads have distinguishable energies — so
    the pipeline uses it as the default and Hungarian as a refinement.
    """
    energies = torch.tensor(
        [head_signature(model, layer, h).sum().item() for h in range(model.cfg.n_heads)]
    )
    return torch.argsort(energies, descending=True)


def canonicalization_residual(model: Transformer, layer: int, head: int) -> float:
    """How nearly degenerate this head's circuit spectrum is, in ``[0, 1]``.

    Near-degenerate singular values mean the canonical basis is ill-conditioned
    and the head's canonicalization carries residual ambiguity.  Reported rather
    than swallowed, since it is a real caveat on any conclusion drawn from
    head-level alignment.
    """
    s = head_signature(model, layer, head)
    s = s[s > s.max() * 1e-6]
    if s.numel() < 2:
        return 0.0
    gaps = (s[:-1] - s[1:]) / s[:-1].clamp_min(1e-12)
    return float(1.0 - gaps.min().clamp(0, 1))
