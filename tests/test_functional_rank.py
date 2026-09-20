"""Functional rank: the behavioral measure that replaced the spectral ones.

The central negative result of Phase 2 is that activation energy does not
measure importance (``docs/FINDINGS.md``). These tests pin the properties that
make the functional rank a usable substitute.
"""

import pytest
import torch

from fusion.capacity.functional import (
    capacity_verdict,
    energy_basis,
    functional_rank,
    restrict_to,
)
from fusion.config import ModelConfig
from fusion.model import Transformer
from fusion.tasks import accuracy, make_tasks


@pytest.fixture
def setup():
    torch.manual_seed(0)
    task = make_tasks(["mod_add"], p=11)[0]
    cfg = ModelConfig(vocab_size=task.vocab.size, d_model=32, n_layers=2, n_heads=4,
                      max_seq_len=task.seq_len)
    model = Transformer(cfg).double()
    tokens, _ = task.batch(256, torch.Generator().manual_seed(1))
    return model, task, tokens


def test_energy_basis_is_orthonormal_and_ordered(setup):
    model, _, tokens = setup
    basis, evals = energy_basis(model, tokens)
    d = model.cfg.d_model
    assert basis.shape == (d, d)
    assert (basis.T @ basis - torch.eye(d, dtype=basis.dtype)).abs().max() < 1e-9
    assert (evals.diff() <= 1e-12).all()
    assert (evals >= 0).all()


def test_restricting_to_the_full_basis_is_a_no_op(setup):
    model, _, tokens = setup
    basis, _ = energy_basis(model, tokens)
    out = restrict_to(model, basis)
    assert (out(tokens) - model(tokens)).abs().max() < 1e-8


def test_restricting_only_touches_writers(setup):
    """The question is how many directions the model needs to *express* itself."""
    model, _, tokens = setup
    basis, _ = energy_basis(model, tokens)
    out = restrict_to(model, basis[:, :8])
    for name, p in out.named_parameters():
        q = dict(model.named_parameters())[name]
        touched = (p.data - q.data).abs().max() > 1e-12
        is_writer = any(name.endswith(w) for w in ("embed", "pos", "w_o", "w_out"))
        if not is_writer:
            assert not touched, f"{name} should not have been projected"


def test_restricting_hard_enough_destroys_the_model(setup):
    model, task, tokens = setup
    basis, _ = energy_basis(model, tokens)
    full = accuracy(model, task, n=512)
    assert accuracy(restrict_to(model, basis[:, :1]), task, n=512) <= full


def test_functional_rank_is_within_bounds_and_self_consistent(setup):
    model, task, tokens = setup
    fr = functional_rank(model, task, tokens, tau=0.99, n_eval=512)
    assert 1 <= fr.rank <= model.cfg.d_model
    assert fr.ratio == fr.rank / model.cfg.d_model
    assert 0.0 <= fr.energy_at_rank <= 1.0 + 1e-9
    assert fr.retained_accuracy >= 0.99 * fr.baseline - 1e-9


def test_a_lower_tau_never_needs_more_directions(setup):
    model, task, tokens = setup
    strict = functional_rank(model, task, tokens, tau=0.99, n_eval=512)
    loose = functional_rank(model, task, tokens, tau=0.50, n_eval=512)
    assert loose.rank <= strict.rank


def test_capacity_verdict_uses_the_behavioral_rank(setup):
    model, task, tokens = setup
    fr = functional_rank(model, task, tokens, n_eval=512)
    d = model.cfg.d_model
    ok, text = capacity_verdict([fr])
    assert ok == (fr.rank <= d)
    assert f"/{d}" in text
    # two copies of the same demand must double the total
    ok2, text2 = capacity_verdict([fr, fr])
    assert ok2 == (2 * fr.rank <= d)
    assert str(2 * fr.rank) in text2


def test_capacity_verdict_handles_no_models():
    ok, text = capacity_verdict([])
    assert not ok and "no models" in text
