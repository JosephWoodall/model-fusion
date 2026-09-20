"""The energy-weighted overlap objective that replaced the hard rank cutoff.

The cutoff versions were untestable in opposite directions (``docs/FINDINGS.md``):
the thresholded rank let its threshold pick the answer, and the participation
ratio was so generous that every configuration passed. These tests pin the
properties that make the weighted objective a usable x-axis instead.
"""

import pytest
import torch

from fusion.capacity.rank import used_subspace
from fusion.capacity.stiefel import (
    KNEE,
    disentangle,
    interference_ratio,
    rearrangement_floor,
    subspace_overlap,
    whiten,
)
from fusion.config import ModelConfig
from fusion.model import Transformer
from fusion.tasks import make_tasks


def orth(d, r, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0][:, :r]


# -- the metric ------------------------------------------------------------


def test_whiten_folds_the_spectrum_in():
    U = orth(16, 4)
    lam = torch.tensor([4.0, 1.0, 0.25, 0.0], dtype=torch.float64)
    M = whiten([U], [lam])[0]
    assert torch.allclose(M, U * lam.sqrt()[None, :])
    # M M^T must reconstruct the covariance
    assert (M @ M.T - (U * lam[None, :]) @ U.T).abs().max() < 1e-12


def test_whiten_rejects_a_mismatched_spectrum():
    with pytest.raises(ValueError, match="energy vector"):
        whiten([orth(16, 4)], [torch.ones(3, dtype=torch.float64)])


def test_weighted_overlap_is_zero_for_orthogonal_ranges():
    Q = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))[0]
    lam = torch.linspace(1.0, 0.1, 4, dtype=torch.float64)
    assert subspace_overlap([Q[:, :4], Q[:, 4:8]], [lam, lam]) < 1e-12


def test_weighted_overlap_is_one_for_proportional_covariances():
    U = orth(16, 4)
    lam = torch.linspace(1.0, 0.1, 4, dtype=torch.float64)
    assert abs(subspace_overlap([U, U], [lam, 3 * lam]) - 1.0) < 1e-9


def test_weighted_overlap_is_bounded():
    """Cauchy-Schwarz on PSD matrices keeps this in [0, 1] whatever the spectra."""
    g = torch.Generator().manual_seed(0)
    for seed in range(5):
        bases = [orth(24, 24, seed=seed), orth(24, 24, seed=seed + 100)]
        energies = [torch.rand(24, generator=g, dtype=torch.float64) for _ in range(2)]
        v = subspace_overlap(bases, energies)
        assert -1e-12 <= v <= 1.0 + 1e-9, v


def test_tail_directions_contribute_in_proportion_to_their_energy():
    """The property the cutoff measures could not have: no cliff at any rank."""
    d = 32
    U = orth(d, d, seed=1)
    strong = torch.zeros(d, dtype=torch.float64)
    strong[:4] = 1.0
    with_tail = strong.clone()
    with_tail[4:] = 1e-6                       # a long, negligible tail
    a = subspace_overlap([U, U], [strong, strong])
    b = subspace_overlap([U, U], [with_tail, with_tail])
    assert abs(a - b) < 1e-4, "a negligible tail must not move the metric"


def test_interference_ratio_takes_the_worst_model_not_the_mean():
    Q = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64))[0]
    lam = torch.ones(4, dtype=torch.float64)
    # models 0 and 1 collide; model 2 is off on its own
    bases = [Q[:, :4], Q[:, :4], Q[:, 8:12]]
    per = interference_ratio(bases, [lam] * 3, per_model=True)
    assert per[2] < 1e-9                       # model 2 sees nothing
    assert per[0] > 0.9                        # models 0 and 1 drown each other
    assert interference_ratio(bases, [lam] * 3) == pytest.approx(max(per))


def test_interference_ratio_of_disjoint_models_is_zero():
    Q = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64))[0]
    lam = torch.linspace(1.0, 0.2, 8, dtype=torch.float64)
    assert interference_ratio([Q[:, :8], Q[:, 8:16]], [lam, lam]) < 1e-12


def test_identical_models_score_n_minus_one():
    """The known false positive, pinned so it cannot be forgotten.

    N identical models merge perfectly but score ``N-1``: the metric cannot tell
    coherent overlap from destructive overlap. Documented in stiefel.py.
    """
    U = orth(16, 4)
    lam = torch.linspace(1.0, 0.1, 4, dtype=torch.float64)
    for n in (2, 3, 4):
        assert interference_ratio([U] * n, [lam] * n) == pytest.approx(n - 1, abs=1e-6)


# -- the floor -------------------------------------------------------------


def test_rearrangement_floor_is_zero_when_a_spectrum_is_rank_deficient():
    """Energy on r << d directions can always be packed away from the others."""
    d = 16
    lam = torch.zeros(d, dtype=torch.float64)
    lam[:4] = 1.0
    assert rearrangement_floor([lam, lam]) < 1e-12


def test_rearrangement_floor_is_positive_for_full_flat_spectra():
    """Two models that fill the stream cannot be separated, whatever the rotation."""
    lam = torch.ones(16, dtype=torch.float64)
    assert rearrangement_floor([lam, lam]) > 0.5


def test_disentangle_cannot_beat_the_floor():
    torch.manual_seed(0)
    d = 24
    bases = [orth(d, d, seed=s) for s in range(2)]
    energies = [torch.linspace(1.0, 0.05, d, dtype=torch.float64) for _ in range(2)]
    res = disentangle(bases, energies, steps=200, lr=0.05)
    assert res.weighted
    assert res.overlap_after >= res.floor - 1e-6, (res.overlap_after, res.floor)


# -- the optimizer ---------------------------------------------------------


def test_weighted_disentangle_reduces_overlap_and_interference():
    torch.manual_seed(0)
    d = 32
    U = orth(d, d, seed=3)
    lam = torch.zeros(d, dtype=torch.float64)
    lam[:6] = torch.linspace(1.0, 0.4, 6, dtype=torch.float64)
    res = disentangle([U.clone(), U.clone()], [lam, lam], steps=300, lr=0.1)
    assert res.overlap_before > 0.9                  # identical to start with
    assert res.overlap_after < 0.05, res.overlap_after
    assert res.interference_after < res.interference_before
    assert res.feasible == (res.interference_after < KNEE)


def test_disentangle_without_energies_is_the_legacy_objective():
    bases = [orth(32, 4, seed=s) for s in range(3)]
    res = disentangle(bases, steps=100, lr=0.1)
    assert not res.weighted
    assert res.feasible is True                       # 12 <= 32
    assert res.interference_after != res.interference_after   # NaN: not measured


def test_result_reports_whether_the_solver_or_capacity_is_the_limit():
    torch.manual_seed(0)
    d = 16
    lam = torch.ones(d, dtype=torch.float64)          # full, flat: unavoidable collision
    res = disentangle([orth(d, d, seed=0), orth(d, d, seed=1)], [lam, lam], steps=200, lr=0.05)
    assert res.floor > 0.5
    assert res.at_floor, (res.overlap_after, res.floor)


# -- integration with a real model ----------------------------------------


def test_full_measure_gives_a_spectrum_aligned_with_its_basis():
    torch.manual_seed(0)
    task = make_tasks(["mod_add"], p=11)[0]
    cfg = ModelConfig(vocab_size=task.vocab.size, d_model=32, n_layers=2, n_heads=4,
                      max_seq_len=task.seq_len)
    model = Transformer(cfg).double()
    tokens, _ = task.batch(128, torch.Generator().manual_seed(0))
    profile = used_subspace(model, tokens, measure="full")
    assert profile.rank == cfg.d_model
    assert abs(profile.energy_captured - 1.0) < 1e-9   # nothing is discarded
    assert profile.basis().shape == (cfg.d_model, cfg.d_model)
    assert profile.spectrum().shape == (cfg.d_model,)
    assert (profile.spectrum().diff() <= 1e-9).all()   # descending


def test_cutoff_profiles_refuse_to_hand_out_a_mismatched_spectrum():
    torch.manual_seed(0)
    task = make_tasks(["mod_add"], p=11)[0]
    cfg = ModelConfig(vocab_size=task.vocab.size, d_model=32, n_layers=2, n_heads=4,
                      max_seq_len=task.seq_len)
    model = Transformer(cfg).double()
    tokens, _ = task.batch(128, torch.Generator().manual_seed(0))
    profile = used_subspace(model, tokens, measure="participation")
    # the pooled spectrum exists, but basis() pools across layers by SVD, so only
    # the "full" path guarantees column-for-column correspondence
    assert profile.rank < cfg.d_model
    assert profile.spectrum().shape[0] == profile.basis().shape[1]
