"""Phase 1: canonicalization must recover a known frame, and must not depend on order."""

import torch

from fusion.align import canonicalize_all
from fusion.align.procrustes import consensus_frame, frame_disagreement, procrustes_rotation
from fusion.config import FusionConfig, ModelConfig
from fusion.model import Transformer
from fusion.symmetry import apply_residual_rotation, random_orthogonal
from fusion.tasks import make_tasks


def build(d=32, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    tasks = make_tasks(["mod_add", "mod_sub"], p=11)
    cfg = ModelConfig(vocab_size=tasks[0].vocab.size, d_model=d, n_layers=2, n_heads=4,
                      max_seq_len=max(t.seq_len for t in tasks))
    return Transformer(cfg).to(dtype), tasks


def test_procrustes_recovers_a_known_rotation():
    torch.manual_seed(0)
    E = torch.randn(64, 16, dtype=torch.float64)
    R = random_orthogonal(16, seed=2, dtype=torch.float64)
    assert (procrustes_rotation(E @ R, E) - R.T).abs().max() < 1e-10


def test_consensus_is_order_invariant():
    torch.manual_seed(0)
    base = torch.randn(64, 16, dtype=torch.float64)
    mats = [base @ random_orthogonal(16, seed=s, dtype=torch.float64) for s in range(4)]
    _, target_a = consensus_frame(mats, iters=50)
    _, target_b = consensus_frame(list(reversed(mats)), iters=50)
    # the consensus is defined up to a global rotation; compare Gram matrices, which are not
    gram_a = target_a @ target_a.T
    gram_b = target_b @ target_b.T
    assert (gram_a - gram_b).abs().max() < 1e-6


def test_canonicalization_undoes_a_pure_frame_difference():
    """Two models identical up to a residual rotation must canonicalize to the same weights."""
    model, _ = build()
    rotated = model.clone()
    apply_residual_rotation(rotated, random_orthogonal(model.cfg.d_model, seed=9,
                                                       dtype=torch.float64))
    before = frame_disagreement([model, rotated])
    canon, report = canonicalize_all([model, rotated], FusionConfig(), refine=False)
    after = frame_disagreement(canon)
    assert after < before * 1e-6, f"{before=} {after=}"
    assert report.improvement() > 0.999


def test_canonicalization_preserves_function():
    model, tasks = build()
    tokens, _ = tasks[0].batch(8, torch.Generator().manual_seed(1))
    other = model.clone()
    apply_residual_rotation(other, random_orthogonal(model.cfg.d_model, seed=4,
                                                     dtype=torch.float64))
    before = [m(tokens).clone() for m in (model, other)]
    canon, _ = canonicalize_all([model, other], FusionConfig(), refine=False)
    for m, b in zip(canon, before, strict=True):
        assert (m(tokens) - b).abs().max() < 1e-7


def test_canonicalization_does_not_mutate_inputs():
    model, _ = build()
    snapshot = [p.detach().clone() for p in model.parameters()]
    other = model.clone()
    canonicalize_all([model, other], FusionConfig())
    for p, s in zip(model.parameters(), snapshot, strict=True):
        assert torch.equal(p.data, s)


def test_canonicalization_scales_to_many_models():
    """N-way alignment is one solve per model, and must not degrade with N."""
    model, _ = build()
    models = [model.clone() for _ in range(6)]
    for i, m in enumerate(models[1:], start=1):
        apply_residual_rotation(m, random_orthogonal(model.cfg.d_model, seed=100 + i,
                                                     dtype=torch.float64))
    canon, report = canonicalize_all(models, FusionConfig(), refine=False)
    assert frame_disagreement(canon) < 1e-12
    assert report.improvement() > 0.999
