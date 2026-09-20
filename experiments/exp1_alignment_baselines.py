"""Phase 1: canonicalization against the published alignment baselines.

Every baseline here searches a strictly smaller group than ``O(d)``: Git
Re-Basin and activation matching search permutations, OT fusion searches soft
permutations, ZipIt! searches feature groupings. None of them can rotate the
residual stream. This experiment measures what that extra freedom is worth.

The comparison is run on a *known* frame difference as well as on independently
trained models, because only the first has a right answer to check against.

    python -m experiments.exp1_alignment_baselines [--steps 1500]
"""

from __future__ import annotations

import argparse

import torch

from fusion.align import canonicalize_all
from fusion.align.baselines import (
    activation_matching,
    git_rebasin_weight_matching,
    ot_fusion,
    zipit_merge,
)
from fusion.capacity.merge import average_models, naive_average
from fusion.config import FusionConfig, ModelConfig, TrainConfig
from fusion.evaluation.metrics import interpolation_barrier
from fusion.symmetry import apply_residual_rotation, random_orthogonal
from fusion.tasks import accuracy, make_tasks
from fusion.train import resolve_device, train_specialist


def evaluate(name: str, merged, task, device, barrier: float | None = None) -> None:
    acc = accuracy(merged, task, device=device)
    extra = f"  barrier {barrier:8.4f}" if barrier is not None else ""
    print(f"  {name:<28} merged acc {acc:.4f}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mod_add")
    ap.add_argument("--p", type=int, default=47)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()

    device = resolve_device()
    task = make_tasks([args.task], p=args.p)[0]
    cfg = ModelConfig(vocab_size=task.vocab.size, d_model=args.d, n_layers=2, n_heads=4,
                      max_seq_len=task.seq_len)
    print(f"device: {device}\nconfig: {cfg}\n")

    specs = [train_specialist(cfg, task, TrainConfig(steps=args.steps, seed=s, log=False))
             for s in (0, 1)]
    models = [s.model for s in specs]
    print(f"specialists: {specs[0].final_accuracy():.4f}, {specs[1].final_accuracy():.4f}")
    tokens, _ = task.batch(512, torch.Generator().manual_seed(0), device=device)

    for label, pair in (
        ("KNOWN FRAME DIFFERENCE (a rotated copy — every method has a right answer to find)",
         None),
        ("INDEPENDENT SEEDS (no method has a right answer to find)", models),
    ):
        print(f"\n=== {label} ===")
        if pair is None:
            rotated = models[0].clone()
            apply_residual_rotation(
                rotated,
                random_orthogonal(args.d, seed=7).to(device=device, dtype=models[0].embed.dtype),
            )
            pair = [models[0], rotated]

        evaluate("naive average (floor)", naive_average(pair), task, device,
                 interpolation_barrier(pair[0], pair[1], task)["barrier"])

        rebasin = [m.clone() for m in pair]
        git_rebasin_weight_matching(rebasin[0], rebasin[1])
        evaluate("git re-basin (weight)", average_models(rebasin), task, device,
                 interpolation_barrier(rebasin[0], rebasin[1], task)["barrier"])

        actmatch = [m.clone() for m in pair]
        activation_matching(actmatch[0], actmatch[1], tokens)
        evaluate("activation matching", average_models(actmatch), task, device,
                 interpolation_barrier(actmatch[0], actmatch[1], task)["barrier"])

        evaluate("OT fusion (sinkhorn)", ot_fusion([m.clone() for m in pair]), task, device)
        evaluate("ZipIt! feature merging", zipit_merge([m.clone() for m in pair], tokens),
                 task, device)

        canon, report = canonicalize_all(pair, FusionConfig(), calib_tokens=tokens)
        evaluate("canonicalization (ours)", average_models(canon), task, device,
                 interpolation_barrier(canon[0], canon[1], task)["barrier"])
        print(f"    {report}")

    print("\nRead this as: on a known frame difference, only a method that can search O(d)")
    print("recovers the answer. On independent seeds, no method recovers anything, because")
    print("there is no frame relating the two models — see experiments/exp0_sanity.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
