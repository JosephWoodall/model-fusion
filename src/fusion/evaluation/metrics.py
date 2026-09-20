"""The diagnostics. Each one is attached to exactly one of the three failure modes.

* :func:`interpolation_barrier` and :func:`layerwise_cka` diagnose **alignment**.
* ``sum r_k / d`` and :func:`fusion.capacity.subspace_overlap` diagnose **capacity**.
* Pre- vs. post-repair accuracy diagnoses **repair**.

:func:`worst_task_accuracy` is the headline accuracy number, not the mean. The
dominant failure mode in N-way merging is one task collapsing to chance while
the others hold, and a mean over N tasks hides exactly that.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..model import Transformer, require_same_arch
from ..tasks import Task, accuracy

# --------------------------------------------------------------------------
# alignment diagnostics
# --------------------------------------------------------------------------


@torch.no_grad()
def interpolation_barrier(
    a: Transformer,
    b: Transformer,
    task: Task,
    n_points: int = 11,
    n_samples: int = 1024,
    metric: str = "loss",
) -> dict:
    """Loss (or error) along the straight line between two models in weight space.

    The barrier is ``max_alpha f(theta_alpha) - lerp(f(theta_0), f(theta_1))``:
    how much worse the worst interpolate is than the straight-line interpolation
    of the endpoints' own values.

    **This is the go/no-go diagnostic.** Run it on the same-task/different-seed
    control.  If aligning two models trained on identical data with different
    seeds does not collapse this toward zero, alignment is broken and no
    downstream number means anything.
    """
    require_same_arch(a, b)
    device = next(a.parameters()).device
    gen = torch.Generator().manual_seed(1234)
    toks, targets = task.batch(n_samples, gen, device=device)

    alphas = torch.linspace(0, 1, n_points).tolist()
    values = []
    probe = a.clone()
    params_a = [p.detach().clone() for p in a.parameters()]
    params_b = [p.detach().clone() for p in b.parameters()]
    for alpha in alphas:
        for p, pa, pb in zip(probe.parameters(), params_a, params_b, strict=True):
            p.data = (1 - alpha) * pa + alpha * pb
        if metric == "loss":
            values.append(probe.loss(toks, targets).item())
        elif metric == "error":
            values.append(1.0 - accuracy(probe, task, n=n_samples, device=device))
        else:
            raise ValueError(f"unknown metric {metric!r}")

    endpoints = [(1 - al) * values[0] + al * values[-1] for al in alphas]
    gaps = [v - e for v, e in zip(values, endpoints, strict=True)]
    return {
        "alphas": alphas,
        "values": values,
        "barrier": max(gaps),
        "argmax_alpha": alphas[max(range(len(gaps)), key=gaps.__getitem__)],
        "metric": metric,
    }


def cka(X: Tensor, Y: Tensor, eps: float = 1e-12) -> float:
    """Linear CKA between two activation matrices ``[n, d]``.

    Invariant to orthogonal transforms and isotropic scaling, which is exactly
    right here: it measures whether two models have the same representational
    *geometry*, independent of the frame Phase 1 is trying to recover.
    """
    X = X.double()
    Y = Y.double()
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = (X.T @ Y).pow(2).sum()
    xtx = (X.T @ X).pow(2).sum().sqrt()
    yty = (Y.T @ Y).pow(2).sum().sqrt()
    return float(xty / (xtx * yty).clamp_min(eps))


@torch.no_grad()
def layerwise_cka(a: Transformer, b: Transformer, tokens: Tensor) -> list[float]:
    """CKA at every residual-stream layer boundary. High and flat = well aligned."""
    from ..capacity.rank import residual_states

    device = next(a.parameters()).device
    tokens = tokens.to(device)
    ha = residual_states(a, tokens)
    hb = residual_states(b.to(device), tokens)
    return [
        cka(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]))
        for x, y in zip(ha, hb, strict=True)
    ]


# --------------------------------------------------------------------------
# accuracy reporting
# --------------------------------------------------------------------------


@torch.no_grad()
def per_task_accuracy(model: Transformer, tasks: list[Task], n: int = 2048) -> dict[str, float]:
    device = next(model.parameters()).device
    return {t.name: accuracy(model, t, n=n, device=device) for t in tasks}


def worst_task_accuracy(accs: dict[str, float]) -> float:
    """The headline number. Means hide single-task collapse."""
    return min(accs.values()) if accs else float("nan")


def normalized_accuracy(
    merged: dict[str, float],
    specialists: dict[str, float],
    chance: dict[str, float] | None = None,
) -> dict[str, float]:
    """Accuracy rescaled so 0 = chance and 1 = the specialist's own accuracy.

    Without this, tasks with different intrinsic difficulty and different chance
    rates cannot be compared or averaged, and the N-way summary is meaningless.
    """
    out = {}
    for name, acc in merged.items():
        ceiling = specialists.get(name, 1.0)
        floor = (chance or {}).get(name, 0.0)
        span = max(ceiling - floor, 1e-9)
        out[name] = (acc - floor) / span
    return out


def chance_rates(tasks: list[Task]) -> dict[str, float]:
    """Exact-match accuracy of uniform random guessing, per task."""
    return {t.name: (1.0 / t.vocab.p) ** t.answer_len for t in tasks}


def summarize(
    merged: dict[str, float],
    specialists: dict[str, float],
    tasks: list[Task],
) -> dict[str, float]:
    """The row that goes into the results table."""
    floors = chance_rates(tasks)
    norm = normalized_accuracy(merged, specialists, floors)
    return {
        "worst_task": worst_task_accuracy(merged),
        "mean_task": sum(merged.values()) / len(merged),
        "worst_normalized": worst_task_accuracy(norm),
        "mean_normalized": sum(norm.values()) / len(norm),
    }
