"""Distilling N specialists into one student of the same size.

This is the competitor merging loses to, and saying so is part of the result.
Distillation gets to use data and gradient descent; merging gets neither. The
defensible claim for merging is not accuracy — it is that it is **data-free and
takes seconds**, and that the capacity law tells you in advance whether it will
work. Reporting distillation is what makes that claim falsifiable rather than
rhetorical.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..config import ModelConfig, TrainConfig
from ..model import Transformer, init_from_seed
from ..tasks import IGNORE, Task, accuracy
from ..train import Specialist, resolve_device


def distill(
    teachers: list[Transformer],
    tasks: list[Task],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig | None = None,
    temperature: float = 2.0,
    hard_weight: float = 0.5,
) -> Specialist:
    """Train one student against N teachers, one teacher per task.

    Loss is the usual mixture: KL to the teacher's softened distribution plus
    the hard-label cross-entropy, both restricted to the answer positions.
    """
    if len(teachers) != len(tasks):
        raise ValueError("need one teacher per task")
    cfg = train_cfg or TrainConfig()
    device = resolve_device(cfg.device)
    student = init_from_seed(model_cfg, cfg.seed + 777, device)
    teachers = [t.to(device).eval() for t in teachers]

    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.98))
    gen = torch.Generator().manual_seed(cfg.seed + 30_000)
    per_task = max(1, cfg.batch_size // len(tasks))
    history: list[dict] = []

    for step in range(cfg.steps):
        student.train()
        for group in opt.param_groups:
            warm = cfg.lr * (step + 1) / cfg.warmup if step < cfg.warmup else None
            t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
            group["lr"] = warm if warm is not None else cfg.lr * 0.5 * (1 + math.cos(math.pi * t))

        loss = torch.zeros((), device=device)
        for teacher, task in zip(teachers, tasks, strict=True):
            toks, targets = task.batch(per_task, gen, device=device)
            mask = (targets != IGNORE).reshape(-1)
            s_logits = student(toks).reshape(-1, model_cfg.vocab_size)[mask]
            with torch.no_grad():
                t_logits = teacher(toks).reshape(-1, model_cfg.vocab_size)[mask]
            soft = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=-1),
                F.log_softmax(t_logits / temperature, dim=-1),
                log_target=True,
                reduction="batchmean",
            ) * temperature**2
            hard = F.cross_entropy(s_logits, targets.reshape(-1)[mask])
            loss = loss + (1 - hard_weight) * soft + hard_weight * hard
        loss = loss / len(tasks)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        if step == cfg.steps - 1 or (cfg.eval_every and step % cfg.eval_every == 0):
            accs = {t.name: accuracy(student, t, n=512, device=device) for t in tasks}
            history.append({"step": step, "loss": loss.item(),
                            "acc": min(accs.values()), "per_task": accs})
            if cfg.log:
                print(f"  [distill] step {step:>6} loss {loss.item():.4f} "
                      f"worst-acc {min(accs.values()):.3f}", flush=True)

    return Specialist(model=student, task=tasks[0], seed=cfg.seed, train_cfg=cfg, history=history)
