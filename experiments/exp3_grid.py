"""Phase 4: the full evaluation grid, with every reference point.

    python -m experiments.exp3_grid --n 2 4 8 --d 128 256 --overlap high medium disjoint

Each cell reports the merge against the floor (naive average), the ceiling (the
specialists), a jointly trained model at the same budget, a router over the N
specialists (the N*P bound), and distillation into one student — the honest
strong competitor, which merging is not expected to beat on accuracy.
"""

from __future__ import annotations

import argparse

from fusion.config import FusionConfig, TrainConfig
from fusion.evaluation.harness import headline_table, run_grid, save


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--d", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--overlap", nargs="+", default=["high", "medium", "disjoint"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--p", type=int, default=47)
    ap.add_argument("--no-distill", action="store_true")
    ap.add_argument("--no-joint", action="store_true")
    ap.add_argument("--out", default="runs/grid.json")
    args = ap.parse_args()

    rows = run_grid(
        n_values=args.n,
        overlaps=args.overlap,
        d_values=args.d,
        out_path=args.out,
        train_cfg=TrainConfig(steps=args.steps, log=False),
        fusion_cfg=FusionConfig(),
        p=args.p,
        include_distill=not args.no_distill,
        include_joint=not args.no_joint,
    )
    save(rows, args.out)
    print("\n" + headline_table(rows))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
