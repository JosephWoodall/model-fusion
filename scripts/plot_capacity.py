"""The headline plot: worst-task accuracy against energy-weighted interference.

    python scripts/plot_capacity.py runs/exp2_weighted_d128.json -o runs/capacity.png

The x-axis is the **interference ratio** reached after disentangling,

    I = max_k sum_{j != k} tr(A_k A_j) / tr(A_k^2)

interference power over signal power in each model's own energy-weighted frame.
The vertical line at 1.0 is the prediction: 0 dB, where interference matches
signal. If the data knees somewhere else, or nowhere, the prediction is wrong,
and drawing the line is how that becomes visible.

The older ``sum_k r_k / d`` x-axis is still plotted for runs made with a cutoff
rank measure, but it is not a defensible axis -- see ``docs/FINDINGS.md``.

Worst-task accuracy, not mean: means hide the single-task collapse this is
looking for. Cells where the solver finished above the rearrangement floor are
drawn hollow, because their overlap reflects the optimizer rather than the
spectra, and nothing about capacity follows from them.
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
            continue
        if "methods" in item:                         # exp3 grid cell
            methods = item["methods"]
            rows.append({
                "ratio": item["capacity_ratio"],
                "interference_after": item.get("subspace_overlap_after"),
                "weighted": False,
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
    return rows


def pick_axis(rows: list[dict]) -> tuple[str, str, float | None]:
    """Interference if the run used the weighted objective, else the legacy ratio."""
    if rows and rows[0].get("weighted"):
        label = r"interference  $\max_k \sum_{j\neq k}$tr$(A_kA_j)\,/\,$tr$(A_k^2)$"
        return "interference_after", label, 1.0
    return "ratio", r"$\sum_k r_k\,/\,d$", 1.0


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

    rows = load_rows(args.path)
    xkey, xlabel, knee = pick_axis(rows)
    rows = sorted([r for r in rows if r.get(xkey) is not None], key=lambda r: r[xkey])
    if not rows:
        print(f"no usable rows in {args.path}")
        return 1

    x = [r[xkey] for r in rows]
    series = [
        ("specialist_worst", "specialists (ceiling)", "-", "o"),
        ("fused_worst_post_repair", "capacity fusion + repair", "-", "s"),
        ("fused_worst_pre_repair", "capacity fusion", "--", "^"),
        ("naive_worst", "naive average (floor)", ":", "x"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, label, style, marker in series:
        ys = [r.get(key) for r in rows]
        if all(y is None for y in ys):
            continue
        ax.plot(x, ys, style, marker=marker, label=label)

    # cells where the solver stopped above the floor say nothing about capacity
    stalled = [(xi, r.get("fused_worst_post_repair"))
               for xi, r in zip(x, rows, strict=True) if r.get("at_floor") is False]
    if stalled:
        ax.scatter([p[0] for p in stalled], [p[1] for p in stalled],
                   s=110, facecolors="none", edgecolors="crimson", lw=1.2, zorder=5,
                   label="solver above floor (not attributable)")

    if knee is not None:
        ax.axvline(knee, color="k", lw=1, alpha=0.6)
        ax.text(knee, 0.55, " predicted knee", rotation=90, va="center",
                fontsize=8, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("worst-task accuracy")
    ax.set_title("Merge degradation against energy-weighted interference")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="center right")
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
