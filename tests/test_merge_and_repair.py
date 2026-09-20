"""Phases 2-3 end to end, plus the reporting invariants that keep results honest."""

import pytest
import torch

from fusion.capacity import fuse, naive_average
from fusion.capacity.merge import average_models
from fusion.config import FusionConfig, ModelConfig
from fusion.evaluation.metrics import (
    chance_rates,
    interpolation_barrier,
    normalized_accuracy,
    worst_task_accuracy,
)
from fusion.model import Transformer, init_from_seed
from fusion.repair import repair_and_report, repair_statistics
from fusion.tasks import make_tasks, max_seq_len


def build(n=2, d=64, p=11):
    tasks = make_tasks(["mod_add", "mod_sub"][:n], p=p)
    cfg = ModelConfig(vocab_size=tasks[0].vocab.size, d_model=d, n_layers=2, n_heads=4,
                      max_seq_len=max_seq_len(tasks))
    models = [init_from_seed(cfg, seed=s) for s in range(n)]
    calib = [t.batch(128, torch.Generator().manual_seed(s))[0] for s, t in enumerate(tasks)]
    return models, tasks, calib, cfg


def test_naive_average_is_the_mean():
    models, _, _, _ = build()
    merged = naive_average(models)
    for name, p in merged.named_parameters():
        want = torch.stack([dict(m.named_parameters())[name].data for m in models]).mean(0)
        assert (p.data - want).abs().max() < 1e-6


def test_merging_mismatched_architectures_is_refused():
    a = Transformer(ModelConfig(vocab_size=20, d_model=32, n_heads=4, max_seq_len=8))
    b = Transformer(ModelConfig(vocab_size=20, d_model=64, n_heads=4, max_seq_len=8))
    with pytest.raises(ValueError, match="architecture mismatch"):
        naive_average([a, b])


def test_average_of_identical_models_is_the_model():
    models, tasks, _, _ = build()
    same = [models[0], models[0].clone()]
    merged = average_models(same)
    tokens, _ = tasks[0].batch(8)
    assert (merged(tokens) - models[0](tokens)).abs().max() < 1e-5


def test_fuse_returns_a_usable_model_and_a_populated_report():
    models, _, calib, cfg = build()
    merged, report = fuse(models, calib, FusionConfig(stiefel_steps=20))
    assert merged.cfg == cfg
    assert report.d_model == cfg.d_model
    assert report.total_rank > 0
    assert report.combine in {"sum", "sum+budget-qp"}
    assert len(report.profiles) == len(models)


def test_fuse_default_is_the_weighted_objective_with_no_cutoff():
    models, _, calib, cfg = build()
    merged, report = fuse(models, calib, FusionConfig(stiefel_steps=20))
    assert report.weighted
    assert report.combine == "sum"                       # no rank budget to spend
    assert all(p.measure == "full" for p in report.profiles)
    assert all(p.rank == cfg.d_model for p in report.profiles)
    assert report.interference_after == report.interference_after   # not NaN
    assert report.floor >= 0.0
    assert report.feasible == (report.interference_after < 1.0)


def test_fuse_with_a_cutoff_measure_still_uses_the_rank_criterion():
    models, _, calib, _ = build()
    _, report = fuse(models, calib, FusionConfig(stiefel_steps=20, rank_measure="participation"))
    assert not report.weighted
    assert report.feasible == (report.total_rank <= report.d_model)


def test_fuse_requires_one_calibration_batch_per_model():
    models, _, calib, _ = build()
    with pytest.raises(ValueError, match="one calibration batch per model"):
        fuse(models, calib[:1], FusionConfig(stiefel_steps=5))


def test_fuse_does_not_mutate_its_inputs():
    models, _, calib, _ = build()
    snapshot = [p.detach().clone() for p in models[0].parameters()]
    fuse(models, calib, FusionConfig(stiefel_steps=10))
    for p, s in zip(models[0].parameters(), snapshot, strict=True):
        assert torch.equal(p.data, s)


def test_repair_attaches_gains_and_folding_is_exact():
    models, tasks, calib, _ = build()
    merged = naive_average(models)
    report = repair_statistics(merged, models, calib[0])
    assert all(n.gain is not None for n in merged.norms())
    assert report.gain_deviation >= 0.0
    tokens, _ = tasks[0].batch(8)
    before = merged(tokens).clone()
    merged.fold_gains()
    assert all(n.gain is None for n in merged.norms())
    assert (merged(tokens) - before).abs().max() < 1e-5


def test_repair_and_report_returns_both_sides_separately():
    """The signature must make it impossible to quote one number for 'after merging'."""
    models, tasks, calib, _ = build()
    merged = naive_average(models)
    before, after, report = repair_and_report(merged, models, tasks, calib[0])
    assert set(before) == set(after) == {t.name for t in tasks}
    assert report.mode == "stats"


def test_worst_task_accuracy_catches_single_task_collapse():
    """The reason the headline metric is the worst task and not the mean."""
    accs = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}
    assert sum(accs.values()) / len(accs) == 0.75      # a mean that looks fine
    assert worst_task_accuracy(accs) == 0.0            # the number that does not


def test_normalized_accuracy_uses_chance_as_the_floor():
    tasks = make_tasks(["mod_add"], p=11)
    floors = chance_rates(tasks)
    assert abs(floors["mod_add"] - 1 / 11) < 1e-12
    norm = normalized_accuracy({"mod_add": 1 / 11}, {"mod_add": 1.0}, floors)
    assert abs(norm["mod_add"]) < 1e-9                 # chance normalizes to zero


def test_interpolation_barrier_is_zero_for_a_model_against_itself():
    models, tasks, _, _ = build()
    out = interpolation_barrier(models[0], models[0].clone(), tasks[0], n_points=5, n_samples=128)
    assert abs(out["barrier"]) < 1e-5
