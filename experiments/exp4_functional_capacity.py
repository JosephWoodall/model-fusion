"""Phase 2, restated: does ``sum_k r_f(tau) <= d`` predict anything?

The energy-weighted interference axis is falsified (``docs/FINDINGS.md``):
energy does not measure importance, so no spectral rank -- thresholded,
participation, or energy-weighted -- sees the directions that decide the
computation. The functional rank asks the behavioral question instead:

    r_f(tau) = smallest r whose top-r directions retain tau of the accuracy

and it saturates with width, so widening the stream makes ``sum_k r_f <= d``
reachable rather than vacuous.

This experiment sweeps ``d`` (and optionally ``N``) across that boundary and
reports worst-task accuracy against the ratio, alongside the three combine
rules so the merge itself is not a confound.

**As of the first run this predicts nothing**: the one feasible cell that beat
chance (d=256, 0.073) does not reproduce at d=512, which is more feasible
still. Single seeds throughout. Run with several ``--seeds`` before drawing a
line through anything.

    python -m experiments.exp4_functional_capacity --d 128 256 512 --n 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fusion.align import canonicalize_all
from fusion.capacity import capacity_verdict, functional_rank, fuse
from fusion.capacity.merge import _sum_models, _wiener_readers
from fusion.config import FusionConfig, ModelConfig, TrainConfig
from fusion.repair import repair_statistics
from fusion.tasks import accuracy, max_seq_len, task_family
from fusion.train import resolve_device, train_family


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--n", type=int, nargs="+", default=[2])
    ap.add_argument("--overlap", default="high", choices=["high", "medium", "disjoint"])
    ap.add_argument("--p", type=int, default=47)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--tau", type=float, default=0.99)
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeats per cell; 1 is an anecdote, not a result")
    ap.add_argument("--out", default="runs/exp4_functional.json")
    args = ap.parse_args()

    dev = resolve_device()
    rows = []
    print(f"{'d':>5} {'N':>3} {'seed':>5} {'sum_rf/d':>9} {'verdict':>14} {'ceiling':>8} "
          f"{'mean':>6} {'wiener':>7} {'wien+rep':>9}")
    print("-" * 78)
    for d in args.d:
      for N in args.n:
       for seed in range(args.seeds):
        tasks = task_family(args.overlap, N, p=args.p)
        cfg = ModelConfig(vocab_size=tasks[0].vocab.size, d_model=d, n_layers=2,
                          n_heads=4, max_seq_len=max_seq_len(tasks))
        specs = train_family(tasks, model_cfg=cfg,
                             train_cfg=TrainConfig(steps=args.steps, log=False),
                             seeds=[seed * 100 + i for i in range(N)])
        models = [s.model for s in specs]
        calib = [t.batch(512, device=dev)[0] for t in tasks]
        frs = [functional_rank(m, t, c, tau=args.tau, n_eval=1024)
               for m, t, c in zip(models, tasks, calib, strict=True)]
        ok, text = capacity_verdict(frs)
        canon, _ = canonicalize_all(models, FusionConfig(), calib_tokens=calib)
        _, cap = fuse(canon, calib, FusionConfig(), canonicalized=True)
        al = cap.aligned

        def worst(model, ts=tasks):
            return min(accuracy(model, t, device=dev) for t in ts)

        wien = _wiener_readers(al, calib)
        rep = wien.clone()
        repair_statistics(rep, al, calib[0])
        total = sum(f.rank for f in frs)
        ceiling = min(s.final_accuracy() for s in specs)
        row = {"d": d, "n": N, "seed": seed, "ratio": total / d, "feasible": ok,
               "verdict": text, "r_f": [f.rank for f in frs], "ceiling": ceiling,
               "naive_worst": worst(_sum_models(al, "mean")), "wiener_worst": worst(wien),
               "wiener_repaired_worst": worst(rep),
               "interference_after": cap.interference_after}
        rows.append(row)
        print(f"{d:>5} {N:>3} {seed:>5} {total/d:>9.2f} "
              f"{'feasible' if ok else 'OVER CAPACITY':>14} {ceiling:>8.3f} "
              f"{row['naive_worst']:>6.3f} {row['wiener_worst']:>7.3f} "
              f"{row['wiener_repaired_worst']:>9.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    feas = [r for r in rows if r["feasible"]]
    over = [r for r in rows if not r["feasible"]]
    if feas and over:
        print(f"  best worst-task, feasible cells     : "
              f"{max(r['wiener_repaired_worst'] for r in feas):.3f}")
        print(f"  best worst-task, over-capacity cells: "
              f"{max(r['wiener_repaired_worst'] for r in over):.3f}")
        print("  The law predicts the first is high and the second is at chance.")
    else:
        print("  Only one side of the boundary is populated; widen --d to test anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
