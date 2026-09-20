"""Training specialists, and the jointly-trained reference model.

Nothing here is novel; it exists so that every model in a fusion experiment is
produced by one code path with one recorded seed.  Provenance matters more than
usual in this project: "different seeds, same data" and "different seeds,
different data" are two different experiments and must not be confused.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .config import ModelConfig, TrainConfig
from .model import Transformer, init_from_seed
from .tasks import Task, accuracy, max_seq_len


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Specialist:
    """A trained model plus everything needed to reproduce and interpret it."""

    model: Transformer
    task: Task
    seed: int
    train_cfg: TrainConfig
    history: list[dict]

    @property
    def name(self) -> str:
        return f"{self.task.name}-s{self.seed}"

    def final_accuracy(self) -> float:
        return self.history[-1]["acc"] if self.history else float("nan")

    def summary(self) -> dict:
        return {
            "name": self.name,
            "task": self.task.name,
            "seed": self.seed,
            "acc": self.final_accuracy(),
            "params": self.model.num_params(),
        }


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * t))


def train_specialist(
    model_cfg: ModelConfig,
    task: Task,
    train_cfg: TrainConfig,
) -> Specialist:
    """Train one model on one task.

    AdamW with high weight decay: the modular-arithmetic tasks are the ones
    whose circuits we want to read in Phase 5, and decay is what makes those
    circuits clean enough to read.
    """
    device = resolve_device(train_cfg.device)
    model = init_from_seed(model_cfg, train_cfg.seed, device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=(0.9, 0.98)
    )
    gen = torch.Generator().manual_seed(train_cfg.seed + 10_000)
    history: list[dict] = []

    for step in range(train_cfg.steps):
        model.train()
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, train_cfg)
        toks, targets = task.batch(train_cfg.batch_size, gen, device=device)
        loss = model.loss(toks, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        last = step == train_cfg.steps - 1
        if last or (train_cfg.eval_every and step % train_cfg.eval_every == 0):
            acc = accuracy(model, task, n=1024, device=device)
            history.append({"step": step, "loss": loss.item(), "acc": acc})
            if train_cfg.log:
                print(
                    f"  [{task.name} s{train_cfg.seed}] step {step:>6} "
                    f"loss {loss.item():.4f}  acc {acc:.3f}",
                    flush=True,
                )

    return Specialist(model=model, task=task, seed=train_cfg.seed, train_cfg=train_cfg,
                      history=history)


def train_family(
    tasks: list[Task],
    d_model: int = 128,
    seeds: list[int] | None = None,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> list[Specialist]:
    """Train one specialist per task, all sharing one architecture and vocabulary."""
    if not tasks:
        raise ValueError("no tasks")
    vocab = tasks[0].vocab
    if any(t.vocab is not vocab for t in tasks):
        raise ValueError("all tasks must share one Vocab instance; use tasks.make_tasks")
    seeds = seeds if seeds is not None else list(range(len(tasks)))
    if len(seeds) != len(tasks):
        raise ValueError("need one seed per task")

    cfg = model_cfg or ModelConfig(
        vocab_size=vocab.size, d_model=d_model, max_seq_len=max_seq_len(tasks)
    )
    base = train_cfg or TrainConfig()
    out = []
    for task, seed in zip(tasks, seeds, strict=True):
        out.append(train_specialist(cfg, task, TrainConfig(**{**asdict(base), "seed": seed})))
    return out


def train_joint(
    tasks: list[Task],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> Specialist:
    """One model trained on all tasks at once — the same-budget reference point.

    This is the number a merge is really competing with, and it is usually
    better than any merge.  Reporting it is not optional.
    """
    device = resolve_device(train_cfg.device)
    model = init_from_seed(model_cfg, train_cfg.seed, device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=(0.9, 0.98)
    )
    gen = torch.Generator().manual_seed(train_cfg.seed + 20_000)
    per_task = max(1, train_cfg.batch_size // len(tasks))
    history: list[dict] = []

    for step in range(train_cfg.steps):
        model.train()
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, train_cfg)
        loss = torch.zeros((), device=device)
        for task in tasks:
            toks, targets = task.batch(per_task, gen, device=device)
            loss = loss + model.loss(toks, targets)
        loss = loss / len(tasks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        last = step == train_cfg.steps - 1
        if last or (train_cfg.eval_every and step % train_cfg.eval_every == 0):
            accs = {t.name: accuracy(model, t, n=512, device=device) for t in tasks}
            history.append({"step": step, "loss": loss.item(),
                            "acc": min(accs.values()), "per_task": accs})
            if train_cfg.log:
                worst = min(accs.values())
                print(f"  [joint] step {step:>6} loss {loss.item():.4f}  worst-acc {worst:.3f}",
                      flush=True)

    return Specialist(model=model, task=tasks[0], seed=train_cfg.seed, train_cfg=train_cfg,
                      history=history)
