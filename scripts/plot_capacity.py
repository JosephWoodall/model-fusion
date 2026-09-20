"""The headline plot: degradation against ``sum_k r_k / d``, with the predicted knee at 1.0.

    python scripts/plot_capacity.py runs/exp2_capacity.json -o runs/capacity.png

Reads the JSON written by ``experiments.exp2_capacity`` (a list of rows) or by
``experiments.exp3_grid`` (a list of cells) and plots worst-task accuracy
against the capacity ratio. Worst-task, not mean: means hide the single-task
collapse this is looking for.

The vertical line at 1.0 is the prediction. If the data knees somewhere else,
the capacity law as stated is wrong, and that is the point of drawing it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    """Accept either experiment's output format."""
    data = json.loads(path.read_text())
    rows = []
    for item in data:
        if "fused_worst_pre_repair" in item:          # exp2
            rows.append(item)
        elif "methods" in item:                       # exp3 grid cell
            methods = item["methods"]
            rows.append({
                "ratio": item["capacity_ratio"],
                "n": item["n_models"],
                "d": item["d_model"],
                "naive_worst": methods.get("naive_average", {}).get("worst_task"),
                "fused_worst_pre_repair":
                    methods.get("capacity_fusion", {}).get("worst_task"),
                "fused_worst_post_repair":
                    methods.get("capacity_fusion_repaired", {}).get("worst_task"),
                "specialist_worst": min(item["specialists"].values())
                if item["specialists"] else None,
            })
    return [r for r in rows if r.get("ratio") is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("runs/capacity.png"))
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed: pip install -e ".[viz]"')
        return 1

    rows = sorted(load_rows(args.path), key=lambda r: r["ratio"])
    if not rows:
        print(f"no usable rows in {args.path}")
        return 1

    x = [r["ratio"] for r in rows]
    series = [
        ("specialist_worst", "specialists (ceiling)", "-", "o"),
        ("fused_worst_post_repair", "capacity fusion + repair", "-", "s"),
        ("fused_worst_pre_repair", "capacity fusion", "--", "^"),
        ("naive_worst", "naive average (floor)", ":", "x"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label, style, marker in series:
        ys = [r.get(key) for r in rows]
        if all(y is None for y in ys):
            continue
        ax.plot(x, ys, style, marker=marker, label=label)

    ax.axvline(1.0, color="k", lw=1, alpha=0.6)
    ax.text(1.02, 0.02, "predicted knee", rotation=90, va="bottom", fontsize=8, alpha=0.7)
    ax.set_xlabel(r"$\sum_k r_k\,/\,d$")
    ax.set_ylabel("worst-task accuracy")
    ax.set_title("Merge degradation against capacity ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    for r, xi in zip(rows, x, strict=True):
        if "n" in r:
            ax.annotate(f"N={r['n']}", (xi, 0.0), fontsize=7, alpha=0.6,
                        xytext=(0, 4), textcoords="offset points", ha="center")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
