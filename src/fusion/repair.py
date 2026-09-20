"""Phase 3 — repair. Fixing the activation statistics of a correct merge.

Even a merge that is right in every other respect has wrong activation scales:
summing or averaging weights changes the variance flowing through the residual
stream, and nothing in the alignment argument controls that.  REPAIR (Jordan et
al.) observes that much of the apparent damage from merging is exactly this, and
that resetting the normalization statistics recovers most of it.

Here the repair surface is deliberately tiny: a gain vector per RMSNorm, a few
thousand parameters, fitted with **no gradient through the body of the network**.
Two modes:

``mode="stats"``
    Data-driven, no optimization: set each gain so the merged model's
    post-norm activation scale matches the average of the specialists'.  This is
    the direct analogue of REPAIR's BatchNorm-statistics reset.

``mode="fit"``
    Fit the gains by gradient descent on calibration data, with every other
    parameter frozen.

**Always report pre- and post-repair numbers separately.**  Repair frequently
recovers most of an apparent failure; conflating the two hides which phase
actually broke, which is the entire point of the three-diagnostic design.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import Transformer
from .tasks import Task, accuracy


@dataclass
class RepairReport:
    mode: str
    steps: int
    gain_deviation: float          # mean |g - 1|: how far off the merge's scales were
    loss_before: float = float("nan")
    loss_after: float = float("nan")

    def __str__(self) -> str:
        return (
            f"repair[{self.mode}] loss {self.loss_before:.4f} -> {self.loss_after:.4f}, "
            f"mean|gain-1| = {self.gain_deviation:.3f}"
        )


@torch.no_grad()
def _norm_input_scales(model: Transformer, tokens: Tensor) -> list[Tensor]:
    """RMS of the input to each RMSNorm, per channel: one ``[d]`` vector per norm."""
    scales: list[Tensor] = []
    hooks = []

    def hook(_mod, args, _out):
        x = args[0].detach()
        scales.append(x.reshape(-1, x.shape[-1]).pow(2).mean(0).sqrt())

    for norm in model.norms():
        hooks.append(norm.register_forward_hook(hook, with_kwargs=False))
    try:
        model.eval()
        model(tokens)
    finally:
        for h in hooks:
            h.remove()
    return scales


@torch.no_grad()
def repair_statistics(
    merged: Transformer,
    references: list[Transformer],
    tokens: Tensor,
    eps: float = 1e-6,
) -> RepairReport:
    """Match the merged model's per-channel activation scale to the specialists' mean.

    No optimization and no labels: one forward pass per model.  This is the
    cheap repair, and on a merge whose only defect is scale it is usually
    enough.
    """
    device = next(merged.parameters()).device
    tokens = tokens.to(device)
    target = [
        torch.stack(s).mean(0)
        for s in zip(*[_norm_input_scales(r.to(device), tokens) for r in references], strict=True)
    ]
    actual = _norm_input_scales(merged, tokens)

    merged.enable_gains()
    deviations = []
    for norm, want, have in zip(merged.norms(), target, actual, strict=True):
        g = (want / have.clamp_min(eps)).clamp(0.1, 10.0)
        norm.gain.data = g.to(norm.gain)
        deviations.append((g - 1).abs().mean().item())
    return RepairReport(
        mode="stats", steps=0, gain_deviation=sum(deviations) / len(deviations)
    )


def repair_fit(
    merged: Transformer,
    tasks: list[Task],
    steps: int = 200,
    lr: float = 0.05,
    batch_size: int = 256,
    seed: int = 0,
) -> RepairReport:
    """Fit only the RMSNorm gains on calibration data. Everything else stays frozen.

    The parameter count is ``(2 * n_layers + 1) * d`` — thousands, not millions —
    so this is not "fine-tuning the merge into working"; it cannot create a
    circuit that the merge destroyed.  That constraint is what keeps the
    post-repair number interpretable.
    """
    device = next(merged.parameters()).device
    for p in merged.parameters():
        p.requires_grad_(False)
    gains = merged.enable_gains()
    for g in gains:
        g.requires_grad_(True)
        g.data = g.data.to(device)

    opt = torch.optim.Adam(gains, lr=lr)
    gen = torch.Generator().manual_seed(seed)

    def batch_loss() -> Tensor:
        loss = torch.zeros((), device=device)
        for task in tasks:
            toks, targets = task.batch(max(1, batch_size // len(tasks)), gen, device=device)
            loss = loss + merged.loss(toks, targets)
        return loss / len(tasks)

    with torch.no_grad():
        before = batch_loss().item()
    for _ in range(steps):
        loss = batch_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = batch_loss().item()

    for p in merged.parameters():
        p.requires_grad_(True)
    dev = sum((g.detach() - 1).abs().mean().item() for g in gains) / len(gains)
    return RepairReport(mode="fit", steps=steps, gain_deviation=dev,
                        loss_before=before, loss_after=after)


def repair_and_report(
    merged: Transformer,
    references: list[Transformer],
    tasks: list[Task],
    tokens: Tensor,
    mode: str = "stats",
    steps: int = 200,
) -> tuple[dict[str, float], dict[str, float], RepairReport]:
    """Evaluate, repair, evaluate again — and return both sets of numbers.

    The return signature forces the pre/post split into the caller's hands so
    that a single "accuracy after merging" number cannot be quoted by accident.
    """
    device = next(merged.parameters()).device
    before = {t.name: accuracy(merged, t, device=device) for t in tasks}
    if mode == "stats":
        report = repair_statistics(merged, references, tokens)
    elif mode == "fit":
        report = repair_fit(merged, tasks, steps=steps)
    else:
        raise ValueError(f"unknown repair mode {mode!r}")
    after = {t.name: accuracy(merged, t, device=device) for t in tasks}
    return before, after, report
