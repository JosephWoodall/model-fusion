"""Spending the residual ``O(d)`` freedom on making the models' used energy disjoint.

After Phase 1 the models share a frame, but they still all pile into the same
high-energy directions.  There is one more degree of freedom available: any
*common* rotation of the merged frame is free, and per-model rotations are free
too as long as each model's rotation is applied to that model before summing.

Two objectives live here.  The second superseded the first.

**Hard-cutoff (legacy).**  ``min sum_{j != k} ||(R_k U_k)^T (R_j U_j)||_F^2``
over a rank-``r_k`` basis ``U_k``.  Zero iff the rotated subspaces are mutually
orthogonal, possible iff ``sum_k r_k <= d``.  This is the original capacity
story and it turned out to be untestable: the answer is set by whatever cutoff
produced ``r_k`` (see ``docs/FINDINGS.md``).  Kept for comparison.

**Energy-weighted (default).**  Drop the cutoff and carry the whole spectrum.
With ``M_k = R_k U_k Lambda_k^{1/2}`` and ``A_k = M_k M_k^T = R_k C_k R_k^T``,

    min over orthogonal R_1..R_N   sum_{j != k} ||M_k^T M_j||_F^2
                                 = sum_{j != k} tr(A_k A_j)

which is the Frobenius inner product between the rotated covariances.  No
threshold appears anywhere: a direction contributes in proportion to the energy
actually on it, so a tail direction carrying 0.1% of the variance contributes
0.1%, rather than being counted as a full dimension or discarded outright.

Three numbers come out of the same traces:

``weighted_overlap``
    ``sum tr(A_k A_j) / sum ||A_k||_F ||A_j||_F``, in ``[0, 1]``.  Bounded and
    symmetric, so it is the x-axis to plot against.

``interference_ratio``
    ``max_k sum_{j != k} tr(A_k A_j) / tr(A_k^2)``.  Interference power over
    signal power in model ``k``'s own energy-weighted frame.  **The predicted
    knee is at 1.0** -- 0 dB, where interference matches signal.  The max rather
    than the mean, to match worst-task accuracy: one model drowning is the
    failure mode, and a mean over N hides it.

``rearrangement_floor``
    The lowest total any rotation could reach, from the rearrangement
    inequality.  Separates "the optimizer stalled" from "this collision is
    unavoidable", which is the actual capacity question.

Optimization is Riemannian gradient descent on the orthogonal manifold with QR
retraction:

1. Euclidean gradient ``G`` from autograd.
2. Project to the tangent space at ``R``:  ``skew(R^T G)``.
3. Step and retract:  ``R <- qf(R - lr * R skew(R^T G))``.

QR retraction rather than Cayley: cheaper, numerically robust at these sizes,
and the fixed points are the same.  ``geoopt`` would do this too; pure torch
keeps the dependency surface small and the step explicit, which matters when
the claim under test is about the optimum.

**Known limitation of the metric.**  ``C_j`` is measured on model ``j``'s own
data.  Inside the merged model, model ``j``'s machinery instead sees model
``k``'s inputs, so the honest interference term is a cross-covariance,
``O(N^2)`` to measure and destructive of the per-model structure.  Overlap here
is therefore a proxy, and it is a *coherent-blind* one: N identical models score
``interference_ratio = N-1`` while merging perfectly.  If a sweep shows no knee
anywhere, this is the first thing to suspect.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _qf(A: Tensor) -> Tensor:
    """QR retraction: the orthogonal factor with a positive-diagonal R."""
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)


def whiten(bases: list[Tensor], energies: list[Tensor] | None) -> list[Tensor]:
    """Fold the spectra into the bases: ``M_k = U_k Lambda_k^{1/2}``.

    With ``energies=None`` this is the identity, which is what recovers the
    legacy hard-cutoff objective: every retained direction gets weight 1 and
    every discarded one weight 0.
    """
    if energies is None:
        return [U.double() for U in bases]
    if len(energies) != len(bases):
        raise ValueError("need one energy vector per basis")
    out = []
    for U, lam in zip(bases, energies, strict=True):
        lam = lam.double().clamp_min(0)[: U.shape[1]]
        if lam.numel() != U.shape[1]:
            raise ValueError(
                f"energy vector has {lam.numel()} entries for a basis of "
                f"{U.shape[1]} columns"
            )
        out.append(U.double() * lam.sqrt()[None, :])
    return out


def _pair_traces(M: list[Tensor]) -> Tensor:
    """``[N, N]`` matrix of ``tr(A_k A_j) = ||M_k^T M_j||_F^2``. Diagonal is ``tr(A_k^2)``."""
    n = len(M)
    out = torch.zeros(n, n, dtype=torch.float64, device=M[0].device)
    for k in range(n):
        for j in range(k, n):
            v = (M[k].T @ M[j]).pow(2).sum()
            out[k, j] = v
            out[j, k] = v
    return out


def subspace_overlap(
    bases: list[Tensor],
    energies: list[Tensor] | None = None,
    normalize: bool = True,
) -> float:
    """Total pairwise overlap between the models' occupied directions.

    Unweighted (``energies=None``): ``sum_{j != k} ||U_k^T U_j||_F^2``, divided
    by ``sum_{j != k} min(r_j, r_k)`` when normalized -- 0 = mutually
    orthogonal, 1 = every pair of subspaces coincides.

    Energy-weighted: ``sum_{j != k} tr(A_k A_j)`` with ``A_k = U_k Lambda_k
    U_k^T``, divided by ``sum_{j != k} ||A_k||_F ||A_j||_F``.  Cauchy-Schwarz
    puts this in ``[0, 1]`` for PSD ``A``, with 1 meaning every pair of rotated
    covariances is proportional.  No cutoff is involved at any point.
    """
    if not bases:
        return float("nan")
    M = whiten(bases, energies)
    T = _pair_traces(M)
    n = len(M)
    off = T.sum() - T.diagonal().sum()
    if not normalize:
        return float(off)
    if energies is None:
        denom = sum(
            min(bases[k].shape[1], bases[j].shape[1])
            for k in range(n) for j in range(n) if j != k
        )
    else:
        norms = T.diagonal().clamp_min(0).sqrt()      # ||A_k||_F
        denom = float((norms[:, None] * norms[None, :]).sum() - norms.pow(2).sum())
    return float(off) / denom if denom > 0 else 0.0


def interference_ratio(
    bases: list[Tensor],
    energies: list[Tensor],
    per_model: bool = False,
) -> float | list[float]:
    """Interference power over signal power, in each model's own weighted frame.

    ``I_k = sum_{j != k} tr(A_k A_j) / tr(A_k^2)``.  Returns ``max_k I_k`` by
    default -- the worst-off model, to match worst-task accuracy, because one
    model drowning is the failure mode and a mean over N hides it.

    **The predicted knee is at 1.0**: interference matches signal, 0 dB.  See
    the module docstring for the two ways that prediction can be wrong.
    """
    M = whiten(bases, energies)
    T = _pair_traces(M)
    diag = T.diagonal()
    per = [
        float((T[k].sum() - diag[k]) / diag[k].clamp_min(1e-30))
        for k in range(len(M))
    ]
    return per if per_model else (max(per) if per else float("nan"))


def spread_initialization(bases: list[Tensor], energies: list[Tensor], d: int) -> list[Tensor]:
    """Warm start that anti-aligns the spectra, which is exactly optimal for N=2.

    Choosing ``R_k = P_k U_k^T`` sends model ``k``'s eigenvectors onto the
    standard axes, permuted by ``P_k``.  The objective then collapses to

        tr(A_k A_j) = sum_i lambda^k_{P_k^-1(i)} lambda^j_{P_j^-1(i)}

    so the problem becomes: lay each model's spectrum along the axes so that
    the models' large eigenvalues land on *different* axes.  For ``N = 2`` the
    rearrangement inequality says reversing one of them is optimal, so this
    start already attains :func:`rearrangement_floor`.  For ``N > 2`` no
    assignment can reverse every pair at once, and staggering the spectra by
    ``d/N`` is the natural generalization -- a strong start, not a proof.

    Riemannian descent then refines from here.  Starting at random instead
    leaves the solver stranded well above the floor, which would confound
    "capacity is the limit" with "the optimizer gave up".
    """
    n = len(bases)
    out = []
    for k, (U, lam) in enumerate(zip(bases, energies, strict=True)):
        order = torch.argsort(lam.double(), descending=True)
        V = U.double()[:, order]                       # eigenvectors, strongest first
        r = V.shape[1]
        if n == 2 and k == 1:
            target = torch.arange(r - 1, -1, -1, device=V.device)   # reverse
        else:
            shift = int(round(k * d / n))
            target = (torch.arange(r, device=V.device) + shift) % max(r, 1)
        P = torch.zeros(r, r, dtype=torch.float64, device=V.device)
        P[target, torch.arange(r, device=V.device)] = 1.0
        # R_k maps V's columns onto the permuted axes; pad to [d, d] if r < d
        R = torch.eye(d, dtype=torch.float64, device=V.device)
        R[:, :r] = V @ P.T
        out.append(_qf(R).T.contiguous())
    return out


def rearrangement_floor(energies: list[Tensor], normalize: bool = True) -> float:
    """Lowest total overlap any set of rotations could reach.

    For a single pair, ``tr(A_k A_j) = sum_{i,i'} lambda^k_i lambda^j_{i'}
    (u_i . v_{i'})^2``, and the matrix of squared inner products is doubly
    stochastic.  Minimizing a bilinear form over the doubly stochastic polytope
    puts the optimum at a permutation, and by the rearrangement inequality the
    minimizing permutation pairs each model's largest eigenvalue with the
    other's smallest.

    Summing that per-pair minimum over all pairs gives a **lower bound** on the
    achievable total: for ``N > 2`` the pairs cannot all be reverse-paired
    simultaneously, so the true floor is at least this.  Compare it against what
    :func:`disentangle` actually reaches -- at the floor the optimizer is done
    and the residual collision is unavoidable, which is the capacity claim.
    """
    n = len(energies)
    lam = [e.double().clamp_min(0).sort(descending=True).values for e in energies]
    total = 0.0
    for k in range(n):
        for j in range(n):
            if j == k:
                continue
            m = min(lam[k].numel(), lam[j].numel())
            total += float((lam[k][:m] * lam[j][:m].flip(0)).sum())
    if not normalize:
        return total
    norms = [float(e.double().clamp_min(0).pow(2).sum().sqrt()) for e in energies]
    denom = sum(norms[k] * norms[j] for k in range(n) for j in range(n) if j != k)
    return total / denom if denom > 0 else 0.0


#: The predicted knee: interference power equal to signal power, 0 dB.
KNEE = 1.0


@dataclass
class DisentangleResult:
    rotations: list[Tensor]
    overlap_before: float
    overlap_after: float
    feasible: bool
    total_rank: int
    d_model: int
    history: list[float]
    weighted: bool = False
    interference_before: float = float("nan")
    interference_after: float = float("nan")
    floor: float = float("nan")

    @property
    def ratio(self) -> float:
        """``sum_k r_k / d`` — the legacy capacity ratio. Meaningless under ``measure="full"``."""
        return self.total_rank / self.d_model

    @property
    def at_floor(self) -> bool:
        """Whether the optimizer reached the unavoidable-collision bound.

        True means the residual overlap is a property of the spectra, not of the
        optimization -- so a merge that fails here fails for capacity reasons.
        False means the solver stalled and the failure is not yet attributable.
        """
        if self.floor != self.floor:
            return False
        return self.overlap_after <= self.floor * 1.05 + 1e-9

    def __str__(self) -> str:
        if not self.weighted:
            verdict = "feasible" if self.feasible else "OVER CAPACITY"
            return (
                f"sum(r_k)/d = {self.total_rank}/{self.d_model} = {self.ratio:.2f} "
                f"({verdict}); overlap {self.overlap_before:.4f} -> {self.overlap_after:.4f}"
            )
        verdict = "below knee" if self.interference_after < KNEE else "ABOVE KNEE"
        stalled = "" if self.at_floor else "  [above floor: solver, not capacity, is the limit]"
        return (
            f"weighted overlap {self.overlap_before:.4f} -> {self.overlap_after:.4f} "
            f"(floor {self.floor:.4f}); interference {self.interference_before:.3f} -> "
            f"{self.interference_after:.3f} ({verdict}){stalled}"
        )


def _disentangle_once(
    bases: list[Tensor],
    energies: list[Tensor] | None = None,
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
    device=None,
    verbose: bool = False,
    init: str = "spread",
) -> DisentangleResult:
    """Find per-model rotations minimizing pairwise overlap of the occupied directions.

    ``bases[k]`` is ``[d, r_k]`` with orthonormal columns.  Returns rotations
    ``R_k`` of shape ``[d, d]``; model ``k``'s rotated subspace is
    ``R_k @ bases[k]``.

    With ``energies[k]`` supplied (one eigenvalue per column of ``bases[k]``)
    the objective is energy-weighted over the **full spectrum** and no rank
    cutoff enters anywhere.  This is the default path from :func:`fusion.fuse`
    under ``rank_measure="full"``.

    Without energies the legacy hard-cutoff objective is used: zero overlap is
    reachable iff ``sum_k r_k <= d``, by a counting argument.

    ``init="spread"`` (the default) warm-starts by anti-aligning the spectra,
    which is exactly optimal for ``N = 2``; ``init="random"`` is the old
    behaviour and is kept so the warm start's contribution can be measured.
    Compare ``overlap_after`` against ``floor`` on the result: above the floor,
    the solver rather than capacity is the binding constraint, and no
    conclusion about capacity follows.
    """
    if not bases:
        raise ValueError("no bases given")
    d = bases[0].shape[0]
    if any(b.shape[0] != d for b in bases):
        raise ValueError("all bases must live in the same d-dimensional stream")
    device = device or bases[0].device
    dtype = torch.float64

    U = [b.detach().to(device=device, dtype=dtype) for b in bases]
    E = None
    if energies is not None:
        E = [e.detach().to(device=device, dtype=dtype) for e in energies]
    total_rank = sum(b.shape[1] for b in U)
    overlap_before = subspace_overlap(U, E)
    interference_before = interference_ratio(U, E) if E is not None else float("nan")
    floor = rearrangement_floor(E) if E is not None else float("nan")

    # fold the spectra in once; the optimizer only ever sees M_k = U_k Lambda_k^{1/2}
    M0 = whiten(U, E)

    if init == "spread" and E is not None:
        R0 = spread_initialization(U, E, d)
    elif init == "spread":
        R0 = [torch.eye(d, dtype=dtype, device=device) for _ in U]
    elif init == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        R0 = [_qf(torch.randn(d, d, generator=g, dtype=dtype)).to(device) for _ in U]
    else:
        raise ValueError(f"unknown initialization {init!r}")
    with torch.enable_grad():
        R = [Rk.to(device=device, dtype=dtype).clone().requires_grad_(True) for Rk in R0]

    history: list[float] = []
    # fuse() runs under no_grad, but this solver needs autograd; ask for it explicitly
    with torch.enable_grad():
        for step in range(steps):
            rotated = [Rk @ Mk for Rk, Mk in zip(R, M0, strict=True)]
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
        rot = [Rk.detach() for Rk in R]
        final = [Rk @ Uk for Rk, Uk in zip(rot, U, strict=True)]
        overlap_after = subspace_overlap(final, E)
        interference_after = interference_ratio(final, E) if E is not None else float("nan")

    out_dtype = bases[0].dtype
    feasible = total_rank <= d if E is None else interference_after < KNEE
    return DisentangleResult(
        rotations=[Rk.to(out_dtype) for Rk in rot],
        overlap_before=overlap_before,
        overlap_after=overlap_after,
        feasible=feasible,
        total_rank=total_rank,
        d_model=d,
        history=history,
        weighted=E is not None,
        interference_before=interference_before,
        interference_after=interference_after,
        floor=floor,
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


def disentangle(
    bases: list[Tensor],
    energies: list[Tensor] | None = None,
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
    device=None,
    verbose: bool = False,
    restarts: int = 3,
) -> DisentangleResult:
    """Minimize pairwise overlap, keeping the best of several initializations.

    The spectral warm start (:func:`spread_initialization`) is exactly optimal
    for ``N = 2`` but, for ``N > 2``, staggering the spectra can start the
    solver in a worse basin than a random rotation would.  Rather than guess,
    run both and keep whichever converges lower -- the objective is cheap
    relative to training the models, and a solver artifact showing up as a
    capacity result is the specific failure this whole phase exists to avoid.

    Check ``result.at_floor`` before reading anything into a high overlap.

    Note the floor is **tight only for N = 2**.  For more models the pairs
    cannot all be reverse-paired at once, so :func:`rearrangement_floor` is a
    lower bound that may not be attainable, and ``at_floor`` being False is then
    weaker evidence that the solver is at fault.
    """
    candidates = [("spread", seed)]
    candidates += [("random", seed + i) for i in range(max(0, restarts - 1))]

    best: DisentangleResult | None = None
    for init, sd in candidates:
        res = _disentangle_once(
            bases, energies, steps=steps, lr=lr, seed=sd,
            device=device, verbose=verbose, init=init,
        )
        if best is None or res.overlap_after < best.overlap_after:
            best = res
        if res.at_floor:                     # provably done; no point continuing
            break
    assert best is not None
    return best
