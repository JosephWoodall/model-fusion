"""The load-bearing tests: every reparameterization must leave the function unchanged.

If any of these fail, no fusion result from this repo means anything — an
accuracy drop could no longer be attributed to the merge rather than to the
change of frame. Everything runs in float64 so the tolerances measure the math
rather than float32 rounding.
"""

import pytest
import torch

from fusion.align.heads import canonical_factors, canonicalize_heads
from fusion.align.mlp import canonicalize_mlp
from fusion.config import ModelConfig
from fusion.model import Transformer
from fusion.symmetry import (
    apply_head_permutation,
    apply_head_transform,
    apply_mlp_permutation,
    apply_mlp_scaling,
    apply_residual_rotation,
    ov_circuit,
    qk_circuit,
    random_orthogonal,
    scramble,
)
from fusion.tasks import make_tasks

TOL = 1e-9


@pytest.fixture
def setup():
    torch.manual_seed(0)
    tasks = make_tasks(["mod_add", "mod_sub"], p=11)
    cfg = ModelConfig(
        vocab_size=tasks[0].vocab.size, d_model=32, n_layers=2, n_heads=4,
        max_seq_len=max(t.seq_len for t in tasks),
    )
    model = Transformer(cfg).double()
    tokens, _ = tasks[0].batch(8, torch.Generator().manual_seed(1))
    return model, tokens, cfg


def _drift(model, tokens, before):
    return (model(tokens) - before).abs().max().item()


def test_residual_rotation_is_exact(setup):
    model, tokens, cfg = setup
    before = model(tokens).clone()
    R = random_orthogonal(cfg.d_model, seed=3, dtype=torch.float64)
    apply_residual_rotation(model, R)
    assert _drift(model, tokens, before) < TOL


def test_residual_rotation_rejects_non_orthogonal(setup):
    model, _, cfg = setup
    with pytest.raises(ValueError, match="not orthogonal"):
        apply_residual_rotation(model, torch.randn(cfg.d_model, cfg.d_model, dtype=torch.float64))


def test_residual_rotation_refuses_active_gains(setup):
    model, _, cfg = setup
    model.enable_gains()
    with pytest.raises(ValueError, match="gains"):
        apply_residual_rotation(model, random_orthogonal(cfg.d_model, dtype=torch.float64))


def test_head_gl_transform_is_exact(setup):
    model, tokens, cfg = setup
    before = model(tokens).clone()
    g = torch.Generator().manual_seed(5)
    for layer in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            A = torch.eye(cfg.d_head, dtype=torch.float64) + 0.3 * torch.randn(
                cfg.d_head, cfg.d_head, generator=g, dtype=torch.float64
            )
            B = torch.eye(cfg.d_head, dtype=torch.float64) + 0.3 * torch.randn(
                cfg.d_head, cfg.d_head, generator=g, dtype=torch.float64
            )
            apply_head_transform(model, layer, h, A=A, B=B)
    assert _drift(model, tokens, before) < 1e-7


def test_head_transform_preserves_circuits(setup):
    model, _, cfg = setup
    qk = qk_circuit(model, 0, 0).clone()
    ov = ov_circuit(model, 0, 0).clone()
    A = torch.eye(cfg.d_head, dtype=torch.float64) + 0.2 * torch.randn(
        cfg.d_head, cfg.d_head, dtype=torch.float64
    )
    apply_head_transform(model, 0, 0, A=A, B=A)
    assert (qk_circuit(model, 0, 0) - qk).abs().max() < 1e-8
    assert (ov_circuit(model, 0, 0) - ov).abs().max() < 1e-8


def test_head_permutation_is_exact(setup):
    model, tokens, cfg = setup
    before = model(tokens).clone()
    apply_head_permutation(model, 0, torch.randperm(cfg.n_heads))
    assert _drift(model, tokens, before) < TOL


def test_mlp_permutation_is_exact(setup):
    model, tokens, cfg = setup
    before = model(tokens).clone()
    apply_mlp_permutation(model, 0, torch.randperm(cfg.ff_dim))
    assert _drift(model, tokens, before) < TOL


def test_mlp_scaling_is_exact_under_relu(setup):
    model, tokens, cfg = setup
    before = model(tokens).clone()
    s = torch.rand(cfg.ff_dim, dtype=torch.float64) + 0.5
    apply_mlp_scaling(model, 0, s)
    assert _drift(model, tokens, before) < TOL


def test_mlp_scaling_refused_under_gelu():
    torch.manual_seed(0)
    tasks = make_tasks(["mod_add"], p=7)
    cfg = ModelConfig(vocab_size=tasks[0].vocab.size, d_model=16, n_heads=2,
                      max_seq_len=tasks[0].seq_len, activation="gelu")
    model = Transformer(cfg).double()
    with pytest.raises(ValueError, match="not function-preserving"):
        apply_mlp_scaling(model, 0, torch.ones(cfg.ff_dim, dtype=torch.float64))


def test_full_scramble_is_exact(setup):
    model, tokens, _ = setup
    before = model(tokens).clone()
    scramble(model, seed=7)
    assert _drift(model, tokens, before) < 1e-7


def test_canonicalize_heads_is_exact(setup):
    model, tokens, _ = setup
    before = model(tokens).clone()
    canonicalize_heads(model)
    assert _drift(model, tokens, before) < 1e-8


def test_canonicalize_mlp_is_exact(setup):
    model, tokens, _ = setup
    before = model(tokens).clone()
    canonicalize_mlp(model)
    assert _drift(model, tokens, before) < TOL


def test_canonical_factors_reconstruct_the_circuit():
    torch.manual_seed(0)
    d, r = 16, 4
    C = torch.randn(d, r, dtype=torch.float64) @ torch.randn(r, d, dtype=torch.float64)
    L, R, _ = canonical_factors(C, r)
    assert (L @ R.T - C).abs().max() < 1e-9


def test_canonicalization_is_gauge_invariant(setup):
    """Two models differing only by the internal gauge must canonicalize to the same weights."""
    model, _, cfg = setup
    other = model.clone()
    g = torch.Generator().manual_seed(11)
    for layer in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            A = torch.eye(cfg.d_head, dtype=torch.float64) + 0.2 * torch.randn(
                cfg.d_head, cfg.d_head, generator=g, dtype=torch.float64
            )
            apply_head_transform(other, layer, h, A=A, B=A)
    canonicalize_heads(model)
    canonicalize_heads(other)
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        assert (a - b).abs().max() < 1e-6


def test_fold_gains_is_exact(setup):
    model, tokens, _ = setup
    model.enable_gains()
    for norm in model.norms():
        norm.gain.data = torch.rand_like(norm.gain) + 0.5
    before = model(tokens).clone()
    model.fold_gains()
    assert all(n.gain is None for n in model.norms())
    assert _drift(model, tokens, before) < TOL
