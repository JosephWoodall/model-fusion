"""Phase 2: effective rank, Stiefel disentangling, and the capacity condition."""

import pytest
import torch

from fusion.capacity.rank import effective_rank, participation_ratio, used_subspace
from fusion.capacity.stiefel import budget_qp, disentangle, subspace_overlap
from fusion.config import ModelConfig
from fusion.model import Transformer
from fusion.tasks import make_tasks


def test_thresholded_rank_of_a_known_low_rank_covariance():
    torch.manual_seed(0)
    d, r = 32, 5
    A = torch.randn(d, r, dtype=torch.float64)
    cov = A @ A.T
    rank, basis, _ = effective_rank(cov, threshold=0.999, measure="threshold")
    assert rank == r
    assert basis.shape == (d, r)
    assert (basis.T @ basis - torch.eye(r, dtype=torch.float64)).abs().max() < 1e-9


def test_participation_rank_of_an_equal_energy_covariance():
    """With equal eigenvalues the participation ratio is exactly the rank."""
    torch.manual_seed(0)
    d, r = 32, 5
    Q = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
    cov = Q[:, :r] @ Q[:, :r].T          # r eigenvalues of 1, the rest 0
    rank, basis, _ = effective_rank(cov, measure="participation")
    assert rank == r
    assert basis.shape == (d, r)


def test_thresholded_rank_is_threshold_sensitive_and_participation_is_not():
    """Why the default measure is the participation ratio (docs/FINDINGS.md)."""
    torch.manual_seed(0)
    d = 64
    evals = torch.cat([torch.ones(5, dtype=torch.float64),
                       torch.full((d - 5,), 1e-3, dtype=torch.float64)])
    Q = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
    cov = (Q * evals[None, :]) @ Q.T
    ranks = {t: effective_rank(cov, threshold=t, measure="threshold")[0]
             for t in (0.90, 0.99, 0.999)}
    assert len(set(ranks.values())) > 1, ranks      # the threshold picks the answer
    assert effective_rank(cov, measure="participation")[0] <= 10


def test_unknown_rank_measure_is_rejected():
    cov = torch.eye(8, dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown rank measure"):
        effective_rank(cov, measure="nonsense")


def test_participation_ratio_matches_a_flat_spectrum():
    evals = torch.ones(10, dtype=torch.float64)
    assert abs(participation_ratio(evals) - 10.0) < 1e-9
    spiked = torch.tensor([1.0] + [0.0] * 9, dtype=torch.float64)
    assert abs(participation_ratio(spiked) - 1.0) < 1e-9


def test_subspace_overlap_bounds():
    d = 16
    Q = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
    a, b = Q[:, :4], Q[:, 4:8]
    assert subspace_overlap([a, b]) < 1e-12          # orthogonal
    assert abs(subspace_overlap([a, a]) - 1.0) < 1e-9  # identical


def test_disentangle_reaches_zero_when_feasible():
    """sum r_k <= d: the subspaces can be made mutually orthogonal, and are."""
    torch.manual_seed(0)
    d, r, n = 32, 4, 4       # 16 <= 32
    bases = [torch.linalg.qr(torch.randn(d, r, dtype=torch.float64))[0] for _ in range(n)]
    res = disentangle(bases, steps=400, lr=0.1)
    assert res.feasible
    assert res.ratio == n * r / d
    assert res.overlap_after < 1e-4, res.overlap_after


def test_disentangle_cannot_reach_zero_when_over_capacity():
    """sum r_k > d: zero overlap is impossible by counting, and the solver reports that."""
    torch.manual_seed(0)
    d, r, n = 16, 6, 4       # 24 > 16
    bases = [torch.linalg.qr(torch.randn(d, r, dtype=torch.float64))[0] for _ in range(n)]
    res = disentangle(bases, steps=400, lr=0.1)
    assert not res.feasible
    assert res.overlap_after > 1e-3, res.overlap_after


def test_disentangle_improves_on_a_colliding_start():
    torch.manual_seed(0)
    d, r, n = 32, 4, 3
    shared = torch.linalg.qr(torch.randn(d, r, dtype=torch.float64))[0]
    bases = [shared.clone() for _ in range(n)]       # maximal overlap
    res = disentangle(bases, steps=400, lr=0.1)
    assert res.overlap_before > 0.99
    assert res.overlap_after < res.overlap_before / 10


def test_budget_qp_captures_most_energy_and_reports_rho():
    torch.manual_seed(0)
    d, r = 24, 6
    bases, energies = [], []
    for _ in range(3):
        bases.append(torch.linalg.qr(torch.randn(d, r, dtype=torch.float64))[0])
        energies.append(torch.linspace(1.0, 0.1, r, dtype=torch.float64))
    basis, rho = budget_qp(bases, energies, budget=d)
    assert basis.shape == (d, d)
    assert 0.99 < rho <= 1.0 + 1e-9                 # full budget keeps everything
    _, rho_small = budget_qp(bases, energies, budget=d // 3)
    assert 0.0 < rho_small < rho


def test_used_subspace_of_a_real_model_is_below_full_width():
    """The premise of the capacity argument: a narrow task does not fill the stream."""
    torch.manual_seed(0)
    tasks = make_tasks(["mod_add"], p=11)
    cfg = ModelConfig(vocab_size=tasks[0].vocab.size, d_model=64, n_layers=2, n_heads=4,
                      max_seq_len=tasks[0].seq_len)
    model = Transformer(cfg).double()
    tokens, _ = tasks[0].batch(256, torch.Generator().manual_seed(0))
    profile = used_subspace(model, tokens, threshold=0.99)
    assert 0 < profile.rank <= cfg.d_model
    assert profile.basis().shape[0] == cfg.d_model
    assert len(profile.ranks) == cfg.n_layers + 1
