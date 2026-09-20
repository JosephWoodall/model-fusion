"""Phase 2: the capacity law, restated without a rank cutoff.

The original claim was ``merge succeeds iff sum_k r_k <= d``. That turned out
to be untestable in both directions (``docs/FINDINGS.md``): a thresholded rank
lets its threshold pick the answer, and the participation ratio is so generous
that every configuration passes while every merge still fails.

Dropping the cutoff, the objective is energy-weighted over the full spectrum,
and the x-axis becomes the **interference ratio**

    I = max_k sum_{j != k} tr(A_k A_j) / tr(A_k^2),      A_k = R_k C_k R_k^T

interference power over signal power in each model's own energy-weighted frame.
The max, not the mean, to match worst-task accuracy.

**The prediction: the knee is at I = 1.0** -- 0 dB, where interference matches
signal. A knee somewhere else, or no knee at all, falsifies it.

The sweep also reports the **rearrangement floor**: the lowest overlap any set
of rotations could reach for these spectra. At the floor the residual collision
is unavoidable and a failure is genuinely a capacity failure; above it the
solver is the limit and nothing is attributable yet.

Pre- and post-repair numbers stay separate, because repair routinely recovers
an apparent capacity failure that was really a scale failure.

    python -m experiments.exp2_capacity [--d 128] [--n 2 3 4 6 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fusion.align import canonicalize_all
from fusion.capacity import fuse, naive_average
from fusion.capacity.stiefel import KNEE
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
    ap.add_argument("--measure", default="full",
                    choices=["full", "participation", "threshold"])
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
            "measure": args.measure,
            "weighted": cap.weighted,
            "feasible": cap.feasible,
            "overlap_before": cap.overlap_before,
            "overlap_after": cap.overlap_after,
            "interference_before": cap.interference_before,
            "interference_after": cap.interference_after,
            "floor": cap.floor,
            "at_floor": cap.at_floor,
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

    weighted = rows[0]["weighted"]
    xlabel = "interference" if weighted else "sum_r/d"
    print(f"{'N':>3} {xlabel:>12} {'floor':>8} {'at_floor':>9} {'overlap':>8} "
          f"{'naive':>7} {'fused':>7} {'repair':>7} {'ceiling':>8}")
    print("-" * 78)
    for r in rows:
        x = r["interference_after"] if weighted else r["ratio"]
        print(f"{r['n']:>3} {x:>12.3f} {r['floor']:>8.4f} {str(r['at_floor']):>9} "
              f"{r['overlap_after']:>8.4f} {r['naive_worst']:>7.3f} "
              f"{r['fused_worst_pre_repair']:>7.3f} {r['fused_worst_post_repair']:>7.3f} "
              f"{r['specialist_worst']:>8.3f}")

    if weighted:
        below = [r for r in rows if r["interference_after"] < KNEE]
        above = [r for r in rows if r["interference_after"] >= KNEE]
        print(f"\n  cells below the predicted knee (I < {KNEE}): {len(below)}")
        print(f"  cells above it                       : {len(above)}")
        if below and above:
            bw = max(r["fused_worst_post_repair"] for r in below)
            aw = max(r["fused_worst_post_repair"] for r in above)
            print(f"  best worst-task accuracy below: {bw:.3f}   above: {aw:.3f}")
            print("  A knee at 1.0 predicts the first is high and the second is at chance.")
        else:
            side = "below" if below else "above"
            print(f"  All cells fall {side} the knee -- this sweep cannot locate it.")
            print("  Widen the sweep until both sides are populated, or the prediction")
            print("  is untested rather than confirmed.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
