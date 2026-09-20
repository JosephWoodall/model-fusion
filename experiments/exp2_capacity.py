"""Phase 2: the capacity law. Does degradation knee at ``sum_k r_k / d = 1``?

Sweeps N at fixed ``d`` so the capacity ratio climbs through 1.0, and reports
worst-task accuracy against it. The prediction is specific and falsifiable:
below the knee the merge should hold near the specialists, above it the
subspaces cannot be made disjoint and accuracy should fall.

Both the pre- and post-repair numbers are printed, because repair routinely
recovers an apparent capacity failure that was really a scale failure, and
conflating them would make the knee look like it is somewhere it is not.

    python -m experiments.exp2_capacity [--d 128] [--n 2 3 4 6 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fusion.align import canonicalize_all
from fusion.capacity import fuse, naive_average
from fusion.config import FusionConfig, ModelConfig, TrainConfig
from fusion.evaluation.metrics import per_task_accuracy, worst_task_accuracy
from fusion.repair import repair_statistics
from fusion.tasks import max_seq_len, task_family
from fusion.train import resolve_device, train_family


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--overlap", default="high", choices=["high", "medium", "disjoint"])
    ap.add_argument("--p", type=int, default=47)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--measure", default="participation", choices=["participation", "threshold"])
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--out", default="runs/exp2_capacity.json")
    args = ap.parse_args()

    device = resolve_device()
    print(f"device: {device}  d={args.d}  overlap={args.overlap}  "
          f"rank measure={args.measure}\n")
    rows = []

    for n in args.n:
        tasks = task_family(args.overlap, n, p=args.p)
        cfg = ModelConfig(vocab_size=tasks[0].vocab.size, d_model=args.d, n_layers=2,
                          n_heads=4, max_seq_len=max_seq_len(tasks))
        fcfg = FusionConfig(rank_threshold=args.threshold, rank_measure=args.measure)
        specs = train_family(tasks, model_cfg=cfg,
                             train_cfg=TrainConfig(steps=args.steps, log=False))
        models = [s.model for s in specs]
        calib = [t.batch(fcfg.calib_batch_size, device=device)[0] for t in tasks]

        canon, _ = canonicalize_all(models, fcfg, calib_tokens=calib)
        merged, cap = fuse(canon, calib, fcfg, canonicalized=True)

        pre = per_task_accuracy(merged, tasks)
        repaired = merged.clone()
        repair_statistics(repaired, models, calib[0])
        post = per_task_accuracy(repaired, tasks)

        row = {
            "n": n,
            "ratio": cap.ratio,
            "total_rank": cap.total_rank,
            "d": args.d,
            "feasible": cap.feasible,
            "overlap_before": cap.overlap_before,
            "overlap_after": cap.overlap_after,
            "captured_energy": cap.captured_energy,
            "specialist_worst": min(s.final_accuracy() for s in specs),
            "naive_worst": worst_task_accuracy(per_task_accuracy(naive_average(models), tasks)),
            "fused_worst_pre_repair": worst_task_accuracy(pre),
            "fused_worst_post_repair": worst_task_accuracy(post),
        }
        rows.append(row)
        print(f"N={n:>2}  {cap}")
        print(f"      specialists {row['specialist_worst']:.3f} | naive {row['naive_worst']:.3f} "
              f"| fused {row['fused_worst_pre_repair']:.3f} "
              f"-> repaired {row['fused_worst_post_repair']:.3f}\n")

    print(f"{'N':>3} {'sum_r/d':>8} {'feasible':>9} {'overlap':>8} "
          f"{'naive':>7} {'fused':>7} {'repair':>7} {'ceiling':>8}")
    print("-" * 64)
    for r in rows:
        print(f"{r['n']:>3} {r['ratio']:>8.2f} {str(r['feasible']):>9} "
              f"{r['overlap_after']:>8.4f} {r['naive_worst']:>7.3f} "
              f"{r['fused_worst_pre_repair']:>7.3f} {r['fused_worst_post_repair']:>7.3f} "
              f"{r['specialist_worst']:>8.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    print("The prediction under test: worst-task accuracy holds while sum_r/d < 1 and")
    print("falls after. A knee anywhere else falsifies the capacity law as stated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
