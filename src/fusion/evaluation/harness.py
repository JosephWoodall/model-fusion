"""The Phase 4 grid: run one cell, collect every reference point, emit one row.

A cell is ``(N, overlap, d_model)``. Running it produces, for the same N
specialists:

    naive average (floor) | canonicalized average | capacity-aware fusion
    | + repair | joint training | router (N*P bound) | distillation

and the diagnostics that say which phase is responsible for whatever gap
appears: interpolation barrier, ``sum r_k / d``, subspace overlap, and the
pre/post-repair split.

Rows are plain dicts so they can be written straight to JSON and plotted
without this module knowing anything about plotting.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..align import canonicalize_all
from ..baselines import Router, distill, router_accuracy
from ..capacity import fuse, naive_average
from ..capacity.merge import average_models
from ..config import FusionConfig, ModelConfig, TrainConfig
from ..model import Transformer
from ..repair import repair_statistics
from ..tasks import max_seq_len, task_family
from ..train import Specialist, train_family, train_joint
from .metrics import interpolation_barrier, per_task_accuracy, summarize


@dataclass
class CellResult:
    n_models: int
    overlap: str
    d_model: int
    specialists: dict[str, float] = field(default_factory=dict)
    methods: dict[str, dict] = field(default_factory=dict)
    capacity_ratio: float = float("nan")
    subspace_overlap_before: float = float("nan")
    subspace_overlap_after: float = float("nan")
    barrier_before: float = float("nan")
    barrier_after: float = float("nan")
    seconds: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def _timed(fn, store: dict, key: str):
    t0 = time.perf_counter()
    out = fn()
    store[key] = time.perf_counter() - t0
    return out


def run_cell(
    n_models: int,
    overlap: str,
    d_model: int,
    train_cfg: TrainConfig | None = None,
    fusion_cfg: FusionConfig | None = None,
    p: int = 47,
    include_distill: bool = True,
    include_joint: bool = True,
    verbose: bool = True,
) -> CellResult:
    """Train N specialists and evaluate every fusion method and reference point."""
    train_cfg = train_cfg or TrainConfig()
    fusion_cfg = fusion_cfg or FusionConfig()
    tasks = task_family(overlap, n_models, p=p)
    vocab = tasks[0].vocab
    model_cfg = ModelConfig(
        vocab_size=vocab.size, d_model=d_model, max_seq_len=max_seq_len(tasks)
    )
    result = CellResult(n_models=n_models, overlap=overlap, d_model=d_model)

    if verbose:
        print(f"[cell] N={n_models} overlap={overlap} d={d_model} "
              f"tasks={[t.name for t in tasks]}", flush=True)

    specs: list[Specialist] = _timed(
        lambda: train_family(tasks, model_cfg=model_cfg, train_cfg=train_cfg,
                             seeds=list(range(n_models))),
        result.seconds, "train_specialists",
    )
    models = [s.model for s in specs]
    device = next(models[0].parameters()).device
    calib = [t.batch(fusion_cfg.calib_batch_size, device=device)[0] for t in tasks]

    # ceiling: each specialist on its own task
    result.specialists = {s.task.name: s.final_accuracy() for s in specs}

    # alignment diagnostic, before and after canonicalization
    result.barrier_before = interpolation_barrier(models[0], models[1], tasks[0])["barrier"] \
        if n_models > 1 else float("nan")

    canon, canon_report = _timed(
        lambda: canonicalize_all(models, fusion_cfg, calib_tokens=calib[0]),
        result.seconds, "canonicalize",
    )
    if n_models > 1:
        result.barrier_after = interpolation_barrier(canon[0], canon[1], tasks[0])["barrier"]
    if verbose:
        print(f"  {canon_report}", flush=True)
        print(f"  barrier {result.barrier_before:.4f} -> {result.barrier_after:.4f}", flush=True)

    def record(name: str, model: Transformer) -> None:
        accs = per_task_accuracy(model, tasks)
        result.methods[name] = {
            "per_task": accs,
            **summarize(accs, result.specialists, tasks),
            "params": model.num_params(),
        }
        if verbose:
            print(f"  {name:<22} worst {result.methods[name]['worst_task']:.3f}  "
                  f"mean {result.methods[name]['mean_task']:.3f}", flush=True)

    record("naive_average", _timed(lambda: naive_average(models), result.seconds, "naive_average"))
    record("canonical_average",
           _timed(lambda: average_models(canon), result.seconds, "canonical_average"))

    merged, cap = _timed(
        lambda: fuse(canon, calib, fusion_cfg, canonicalized=True),
        result.seconds, "fuse",
    )
    result.capacity_ratio = cap.ratio
    result.subspace_overlap_before = cap.overlap_before
    result.subspace_overlap_after = cap.overlap_after
    if verbose:
        print(f"  {cap}", flush=True)
    record("capacity_fusion", merged)

    repaired = merged.clone()
    rep = _timed(lambda: repair_statistics(repaired, models, calib[0]),
                 result.seconds, "repair")
    record("capacity_fusion_repaired", repaired)
    result.methods["capacity_fusion_repaired"]["repair"] = asdict(rep)

    # reference points
    router = Router({t.name: m for t, m in zip(tasks, models, strict=True)})
    accs = router_accuracy(router, tasks)
    result.methods["router"] = {
        "per_task": accs, **summarize(accs, result.specialists, tasks),
        "params": router.num_params(),
    }

    if include_joint:
        joint = _timed(lambda: train_joint(tasks, model_cfg, train_cfg),
                       result.seconds, "train_joint")
        record("joint_training", joint.model)

    if include_distill:
        student = _timed(lambda: distill(models, tasks, model_cfg, train_cfg),
                         result.seconds, "distill")
        record("distillation", student.model)

    return result


def run_grid(
    n_values: list[int],
    overlaps: list[str],
    d_values: list[int],
    out_path: str | Path | None = None,
    **kw,
) -> list[CellResult]:
    """The full Phase 4 grid. Writes incrementally so a long run is never lost."""
    rows: list[CellResult] = []
    path = Path(out_path) if out_path else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
    for d in d_values:
        for overlap in overlaps:
            for n in n_values:
                rows.append(run_cell(n, overlap, d, **kw))
                if path:
                    path.write_text(json.dumps([r.to_json() for r in rows], indent=2))
    return rows


def headline_table(rows: list[CellResult]) -> str:
    """Degradation against ``sum r_k / d`` — the plot, as text.

    The prediction under test is a knee at ratio 1.0: cells below it should keep
    worst-task accuracy near the specialists', cells above it should fall off.
    """
    lines = [
        f"{'N':>3} {'overlap':>9} {'d':>5} {'sum_r/d':>8} {'overlap':>8} "
        f"{'naive':>7} {'canon':>7} {'fused':>7} {'repair':>7} {'distil':>7}",
        "-" * 82,
    ]
    def worst(row: CellResult, name: str) -> str:
        m = row.methods.get(name)
        return f"{m['worst_task']:.3f}" if m else "   -  "

    def ratio(row: CellResult) -> float:
        # NaN sorts unpredictably; park unmeasured cells at the front
        return row.capacity_ratio if row.capacity_ratio == row.capacity_ratio else 0.0

    for r in sorted(rows, key=ratio):
        lines.append(
            f"{r.n_models:>3} {r.overlap:>9} {r.d_model:>5} {r.capacity_ratio:>8.2f} "
            f"{r.subspace_overlap_after:>8.4f} {worst(r, 'naive_average'):>7} "
            f"{worst(r, 'canonical_average'):>7} {worst(r, 'capacity_fusion'):>7} "
            f"{worst(r, 'capacity_fusion_repaired'):>7} {worst(r, 'distillation'):>7}"
        )
    return "\n".join(lines)


def save(rows: list[CellResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_json() for r in rows], indent=2))


@torch.no_grad()
def load(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())
