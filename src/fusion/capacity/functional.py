"""Functional rank: how many directions a model actually *needs*, by ablation.

Every spectral rank in :mod:`fusion.capacity.rank` answers "where is the
activation energy?".  That turned out to be the wrong question.  Measured on
this testbed at ``d=128``, a grokked modular-addition model:

===========  =============  ========
directions   energy kept    accuracy
===========  =============  ========
64           92.7%          0.48
80           95.7%          0.82
96           97.8%          0.99
112          99.1%          1.00
===========  =============  ========

Ninety-three percent of the energy is worth less than half the accuracy.  The
last one percent, spread over roughly forty directions, carries the rest.  A
low-variance direction can be functionally critical -- it takes very little
energy to move a decision boundary -- so **activation energy is not a measure of
importance**, and every capacity statement built on a spectral rank inherits
that error.

The functional rank asks the behavioral question instead:

    r_f(tau) = the smallest r such that confining the model's writes to its top
               r directions retains at least ``tau`` of its original accuracy.

``tau`` is a knob, but a *behavioral* one -- "how much accuracy am I willing to
lose" is a question with a meaningful answer, unlike "what eigenvalue counts as
zero".  And unlike the spectral ranks, it is measured against the thing the
capacity law is actually about.

This costs ``O(log d)`` evaluations per model via bisection, which is why it is
a measurement rather than something inside an optimization loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..model import Transformer
from ..tasks import Task, accuracy
from .rank import read_point_covariance


@dataclass
class FunctionalRank:
    """What a model needs, versus where its energy is."""

    rank: int
    d_model: int
    tau: float
    baseline: float
    retained_accuracy: float
    energy_at_rank: float

    @property
    def ratio(self) -> float:
        """``r_f / d`` — the per-model share of the residual stream."""
        return self.rank / self.d_model

    def __str__(self) -> str:
        return (
            f"r_f({self.tau}) = {self.rank}/{self.d_model} = {self.ratio:.3f} "
            f"(acc {self.retained_accuracy:.3f} of {self.baseline:.3f}; "
            f"only {100 * self.energy_at_rank:.2f}% of energy)"
        )


@torch.no_grad()
def energy_basis(model: Transformer, tokens: Tensor) -> tuple[Tensor, Tensor]:
    """Pooled second-moment eigenbasis of the residual stream, strongest first.

    Uncentered, and pooled over read points with each trace-normalized so no
    single point dominates — the same convention the merge projectors use.
    """
    covs = read_point_covariance(model, tokens, center=False)
    pooled = sum(c / c.diagonal().sum().clamp_min(1e-30) for c in covs) / len(covs)
    evals, evecs = torch.linalg.eigh(pooled)
    order = torch.argsort(evals, descending=True)
    return evecs[:, order], evals[order].clamp_min(0)


@torch.no_grad()
def restrict_to(model: Transformer, basis: Tensor) -> Transformer:
    """A copy of ``model`` whose every write to the residual stream lands in ``basis``.

    Only writers are projected.  Readers are left alone deliberately: the
    question is how many directions the model needs to *express* its
    computation, and projecting readers too would conflate that with how
    robustly it reads.
    """
    P = (basis.double() @ basis.double().T).to(model.embed.dtype)
    out = model.clone()
    out.embed.data = out.embed.data @ P
    out.pos.data = out.pos.data @ P
    for block in out.blocks:
        block.attn.w_o.data = block.attn.w_o.data @ P
        block.mlp.w_out.data = block.mlp.w_out.data @ P
    return out


@torch.no_grad()
def functional_rank(
    model: Transformer,
    task: Task,
    tokens: Tensor,
    tau: float = 0.99,
    n_eval: int = 2048,
) -> FunctionalRank:
    """Smallest number of directions that retains ``tau`` of the model's accuracy.

    Found by bisection, which assumes accuracy is monotone in ``r``.  It is not
    exactly monotone -- the curve is noisy near the transition -- so treat the
    result as accurate to a few directions, not exactly.
    """
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    d = model.cfg.d_model
    basis, evals = energy_basis(model, tokens)
    cum = torch.cumsum(evals, 0) / evals.sum().clamp_min(1e-30)
    baseline = accuracy(model, task, n=n_eval, device=device)

    def acc_at(r: int) -> float:
        return accuracy(restrict_to(model, basis[:, :r]), task, n=n_eval, device=device)

    lo, hi = 1, d
    while lo < hi:
        mid = (lo + hi) // 2
        if acc_at(mid) >= tau * baseline:
            hi = mid
        else:
            lo = mid + 1
    return FunctionalRank(
        rank=lo, d_model=d, tau=tau, baseline=baseline,
        retained_accuracy=acc_at(lo), energy_at_rank=float(cum[lo - 1]),
    )


def capacity_verdict(ranks: list[FunctionalRank]) -> tuple[bool, str]:
    """Does ``sum_k r_f(k) <= d`` hold, and what does that predict?

    This is the capacity law restated over a rank that is defined behaviorally
    rather than spectrally.  It is the version worth testing; whether it holds
    is a separate question from whether it is well posed, and only this version
    is well posed.
    """
    if not ranks:
        return False, "no models"
    d = ranks[0].d_model
    total = sum(r.rank for r in ranks)
    ok = total <= d
    verdict = "feasible" if ok else "OVER CAPACITY"
    return ok, (
        f"sum(r_f)/d = {total}/{d} = {total / d:.2f} ({verdict}); "
        f"per-model r_f/d = {[round(r.ratio, 3) for r in ranks]}"
    )
