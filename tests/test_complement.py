"""The rank-deficiency of the embedding anchor, and the fix for it.

These tests encode the central Phase 1 finding (``docs/FINDINGS.md``): Procrustes
on the shared-vocabulary matrices pins the residual frame only on their span,
and whenever ``d`` exceeds that span the rest of the frame is arbitrary.
"""

import torch

from fusion.align import canonicalize_all
from fusion.align.complement import (
    anchor_basis,
    anchor_matrix,
    complement_conditioning,
    complete_residual_frame_,
    orthogonal_complement,
    pooled_covariance,
)
from fusion.align.procrustes import align_residual_, frame_disagreement
from fusion.config import FusionConfig, ModelConfig
from fusion.model import Transformer
from fusion.symmetry import apply_residual_rotation, random_orthogonal
from fusion.tasks import make_tasks


def build(d=64, p=11, dtype=torch.float64):
    torch.manual_seed(0)
    task = make_tasks(["mod_add"], p=p)[0]
    cfg = ModelConfig(vocab_size=task.vocab.size, d_model=d, n_layers=2, n_heads=4,
                      max_seq_len=task.seq_len)
    model = Transformer(cfg).to(dtype)
    tokens, _ = task.batch(256, torch.Generator().manual_seed(1))
    return model, task, tokens


def test_anchor_is_rank_deficient_when_d_exceeds_vocab():
    """The premise of the whole complement fix."""
    model, _, _ = build(d=64, p=11)          # V = 11 + 3 + 1 = 15, T = 6
    A = anchor_matrix(model, include_unembed=False)
    assert A.shape[0] < model.cfg.d_model
    assert torch.linalg.matrix_rank(A.T.double()).item() == A.shape[0]


def test_unembedding_raises_the_anchor_rank():
    """U^T is also in the shared token basis -- free extra rank, so use it."""
    model, _, _ = build(d=64, p=11)
    without = anchor_matrix(model, include_unembed=False).shape[0]
    with_u = anchor_matrix(model, include_unembed=True).shape[0]
    assert with_u > without


def test_orthogonal_complement_is_orthonormal_and_complementary():
    torch.manual_seed(0)
    d, r = 32, 12
    basis = torch.linalg.qr(torch.randn(d, r, dtype=torch.float64))[0]
    Q = orthogonal_complement(basis)
    assert Q.shape == (d, d - r)
    assert (Q.T @ Q - torch.eye(d - r, dtype=torch.float64)).abs().max() < 1e-10
    assert (basis.T @ Q).abs().max() < 1e-10


def test_procrustes_alone_leaves_the_complement_arbitrary():
    """Embedding disagreement goes to zero while the weights still differ."""
    model, _, _ = build(d=64, p=11)
    rotated = model.clone()
    apply_residual_rotation(rotated, random_orthogonal(64, seed=5, dtype=torch.float64))
    pair = [model.clone(), rotated.clone()]
    align_residual_(pair)
    assert frame_disagreement(pair) < 1e-20          # the anchor is perfectly aligned...
    worst = max((a.data - b.data).abs().max().item()
                for a, b in zip(pair[0].parameters(), pair[1].parameters(), strict=True))
    assert worst > 1e-3, "expected the complement to remain misaligned"


def test_complement_fix_recovers_a_planted_rotation():
    """With the complement pinned, a known rotation is recovered to numerical precision."""
    model, _, tokens = build(d=64, p=11)
    rotated = model.clone()
    apply_residual_rotation(rotated, random_orthogonal(64, seed=5, dtype=torch.float64))
    pair = [model.clone(), rotated.clone()]
    align_residual_(pair)
    complete_residual_frame_(pair, tokens)
    worst = max((a.data - b.data).abs().max().item()
                for a, b in zip(pair[0].parameters(), pair[1].parameters(), strict=True))
    assert worst < 1e-6, worst


def test_full_pipeline_recovers_a_planted_rotation():
    model, task, tokens = build(d=64, p=11)
    rotated = model.clone()
    apply_residual_rotation(rotated, random_orthogonal(64, seed=13, dtype=torch.float64))
    canon, report = canonicalize_all([model, rotated], FusionConfig(), calib_tokens=tokens)
    assert report.complement_fixed
    worst = max((a.data - b.data).abs().max().item()
                for a, b in zip(canon[0].parameters(), canon[1].parameters(), strict=True))
    assert worst < 1e-5, worst
    # and the function is untouched
    before = model(tokens)
    assert (canon[0](tokens) - before).abs().max() < 1e-6


def test_complement_canonicalization_is_function_preserving():
    model, _, tokens = build(d=64, p=11)
    other = model.clone()
    before = model(tokens).clone()
    pair = [model, other]
    align_residual_(pair)
    complete_residual_frame_(pair, tokens)
    assert (pair[0](tokens) - before).abs().max() < 1e-7


def test_conditioning_report_is_populated():
    model, _, tokens = build(d=64, p=11)
    basis = anchor_basis([model])
    cond = complement_conditioning(model, tokens, basis)
    assert cond["complement_dim"] == model.cfg.d_model - basis.shape[1]
    assert 0.0 <= cond["energy_fraction"] <= 1.0
    assert 0.0 <= cond["min_relative_gap"] <= 1.0


def test_pooled_covariance_is_symmetric_psd():
    model, _, tokens = build(d=64, p=11)
    cov = pooled_covariance(model, tokens)
    assert (cov - cov.T).abs().max() < 1e-10
    assert torch.linalg.eigvalsh(cov).min() > -1e-8
