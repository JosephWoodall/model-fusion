"""Spending the residual ``O(d)`` freedom on making the models' used subspaces disjoint.

After Phase 1 the models share a frame, but they still all pile into the same
few high-energy directions.  There is one more degree of freedom available: any
*common* rotation of the merged frame is free, and per-model rotations are free
too as long as we apply each model's rotation to that model before summing.

So solve

    min over orthogonal R_1..R_N   sum_{j != k} || (R_k U_k)^T (R_j U_j) ||_F^2

where ``U_k`` is model ``k``'s used-subspace basis.  The objective is exactly
the total pairwise subspace overlap: it is zero iff the rotated subspaces are
mutually orthogonal, which is possible iff ``sum_k r_k <= d``.

Optimization is on the Stiefel/orthogonal manifold by Riemannian gradient
descent with QR retraction:

1. Euclidean gradient ``G`` from autograd.
2. Project to the tangent space at ``R``:  ``skew(R^T G)``.
3. Step and retract:  ``R <- qf(R - lr * R skew(R^T G))``.

QR retraction rather than Cayley: it is cheaper, numerically robust at these
sizes, and the fixed points are the same.  ``geoopt`` would do this too, but a
pure-torch implementation keeps the dependency surface small and the step
explicit, which matters when the claim being tested is about the optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _qf(A: Tensor) -> Tensor:
    """QR retraction: the orthogonal factor with a positive-diagonal R."""
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)


def subspace_overlap(bases: list[Tensor], normalize: bool = True) -> float:
    """Total pairwise overlap ``sum_{j != k} ||U_k^T U_j||_F^2``.

    With ``normalize=True`` the result is divided by ``sum_{j != k} min(r_j, r_k)``,
    its maximum, giving a number in ``[0, 1]``: 0 = mutually orthogonal,
    1 = every pair of subspaces coincides.
    """
    total, denom = 0.0, 0.0
    for k, Uk in enumerate(bases):
        for j, Uj in enumerate(bases):
            if j == k:
                continue
            total += (Uk.T.double() @ Uj.double()).pow(2).sum().item()
            denom += min(Uk.shape[1], Uj.shape[1])
    if not normalize or denom == 0:
        return total
    return total / denom


@dataclass
class DisentangleResult:
    rotations: list[Tensor]
    overlap_before: float
    overlap_after: float
    feasible: bool
    total_rank: int
    d_model: int
    history: list[float]

    @property
    def ratio(self) -> float:
        """``sum_k r_k / d`` — the capacity ratio the headline plot is against."""
        return self.total_rank / self.d_model

    def __str__(self) -> str:
        verdict = "feasible" if self.feasible else "OVER CAPACITY"
        return (
            f"sum(r_k)/d = {self.total_rank}/{self.d_model} = {self.ratio:.2f} ({verdict}); "
            f"overlap {self.overlap_before:.4f} -> {self.overlap_after:.4f}"
        )


def disentangle(
    bases: list[Tensor],
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
    device=None,
    verbose: bool = False,
) -> DisentangleResult:
    """Find per-model rotations minimizing pairwise subspace overlap.

    ``bases[k]`` is ``[d, r_k]`` with orthonormal columns.  Returns rotations
    ``R_k`` of shape ``[d, d]``; the rotated subspace of model ``k`` is
    ``R_k @ bases[k]``.

    When ``sum_k r_k <= d`` the global optimum is zero overlap, and in practice
    this converges to it.  When ``sum_k r_k > d`` zero is unattainable by a
    counting argument, and the converged overlap measures the unavoidable
    collision — which is the quantity the capacity law predicts degradation
    from.
    """
    if not bases:
        raise ValueError("no bases given")
    d = bases[0].shape[0]
    if any(b.shape[0] != d for b in bases):
        raise ValueError("all bases must live in the same d-dimensional stream")
    device = device or bases[0].device
    dtype = torch.float64

    U = [b.detach().to(device=device, dtype=dtype) for b in bases]
    total_rank = sum(b.shape[1] for b in U)
    overlap_before = subspace_overlap(U)

    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.enable_grad():
        R = [
            _qf(torch.randn(d, d, generator=g, dtype=dtype)).to(device).requires_grad_(True)
            for _ in U
        ]

    history: list[float] = []
    # fuse() runs under no_grad, but this solver needs autograd; ask for it explicitly
    with torch.enable_grad():
        for step in range(steps):
            rotated = [Rk @ Uk for Rk, Uk in zip(R, U, strict=True)]
            loss = torch.zeros((), dtype=dtype, device=device)
            for k in range(len(rotated)):
                for j in range(k + 1, len(rotated)):
                    loss = loss + (rotated[k].T @ rotated[j]).pow(2).sum()
            loss = 2 * loss  # both orderings of each pair
            history.append(loss.item())

            grads = torch.autograd.grad(loss, R)
            with torch.no_grad():
                for Rk, G in zip(R, grads, strict=True):
                    A = Rk.T @ G
                    tangent = Rk @ (A - A.T) / 2      # project onto the tangent space
                    Rk.copy_(_qf(Rk - lr * tangent))   # step, then retract
            if verbose and step % max(1, steps // 10) == 0:
                print(f"    stiefel step {step:>4}  overlap-loss {loss.item():.6e}", flush=True)

    with torch.no_grad():
        final = [Rk.detach() @ Uk for Rk, Uk in zip(R, U, strict=True)]
        overlap_after = subspace_overlap(final)

    out_dtype = bases[0].dtype
    return DisentangleResult(
        rotations=[Rk.detach().to(out_dtype) for Rk in R],
        overlap_before=overlap_before,
        overlap_after=overlap_after,
        feasible=total_rank <= d,
        total_rank=total_rank,
        d_model=d,
        history=history,
    )


def budget_qp(
    bases: list[Tensor],
    energies: list[Tensor],
    budget: int,
) -> tuple[Tensor, float]:
    """Choose ``budget`` directions out of the union of used subspaces.

    This is the over-capacity branch: when ``sum_k r_k > d``, something must be
    discarded, and this picks what.  The selection maximizes retained activation
    energy, which is the output-space objective in closed form — for an
    orthogonality-constrained basis the QP's optimum is the top eigenvectors of
    the energy-weighted scatter matrix.

    Returns ``(basis[d, budget], rho)`` where ``rho`` is the *captured energy*:
    the fraction of total activation energy the chosen subspace retains.  ``rho``
    is the honest report of how much the merge had to throw away.
    """
    if len(bases) != len(energies):
        raise ValueError("need one energy vector per basis")
    d = bases[0].shape[0]
    budget = min(budget, d)
    M = torch.zeros(d, d, dtype=torch.float64, device=bases[0].device)
    total = 0.0
    for U, e in zip(bases, energies, strict=True):
        w = e[: U.shape[1]].double().clamp_min(0)
        M += (U.double() * w[None, :]) @ U.double().T
        total += w.sum().item()
    evals, evecs = torch.linalg.eigh(M)
    order = torch.argsort(evals, descending=True)
    kept = evals[order][:budget].clamp_min(0).sum().item()
    rho = kept / total if total > 0 else float("nan")
    return evecs[:, order[:budget]].to(bases[0].dtype), rho
