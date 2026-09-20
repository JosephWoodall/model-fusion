"""How N models are actually combined once their geometry is disentangled.

Phase 2 spends a lot of effort making the models' occupied directions
orthogonal. None of that pays off unless each model's *readers* then see its own
signal at the right magnitude, and neither fixed rule achieves that:

* averaging the readers scales each by ``1/N`` while the stream keeps its full
  magnitude, so attention scores shrink by ``N^2`` and the softmax flattens;
* summing them has the mirror problem once the models genuinely overlap.

The two degenerate merges below pin this down, because both have a known right
answer. Merging a model with a **zero** model must be a no-op, and merging it
with a **copy of itself** must also be a no-op. No fixed rule passes both; the
Wiener rule passes both by construction.
"""

import pytest
import torch

from fusion.capacity.merge import _sum_models, _wiener_readers
from fusion.capacity.rank import read_point_covariance, read_point_states
from fusion.config import ModelConfig
from fusion.model import Transformer
from fusion.tasks import make_tasks


@pytest.fixture
def setup():
    torch.manual_seed(0)
    task = make_tasks(["mod_add"], p=11)[0]
    cfg = ModelConfig(vocab_size=task.vocab.size, d_model=32, n_layers=2, n_heads=4,
                      max_seq_len=task.seq_len)
    model = Transformer(cfg).double()
    tokens, _ = task.batch(256, torch.Generator().manual_seed(1))
    return model, task, tokens


def zeroed(model):
    out = model.clone()
    for p in out.parameters():
        p.data.zero_()
    return out


def _logit_gap(a, b, tokens):
    """Max absolute logit difference, ignoring a global scale on the logits."""
    la, lb = a(tokens), b(tokens)
    scale = (la * lb).sum() / lb.pow(2).sum().clamp_min(1e-30)
    return (la - scale * lb).abs().max().item()


# -- read points -----------------------------------------------------------


def test_read_points_count_matches_the_norm_layers(setup):
    model, _, tokens = setup
    states = read_point_states(model, tokens)
    assert len(states) == 2 * model.cfg.n_layers + 1 == len(model.norms())
    assert all(h.shape[-1] == model.cfg.d_model for h in states)


def test_attention_and_mlp_of_one_layer_read_different_points(setup):
    """Why a per-layer-boundary covariance is the wrong projector for half the readers."""
    model, _, tokens = setup
    states = read_point_states(model, tokens)
    assert not torch.allclose(states[0], states[1])


def test_read_point_covariance_can_skip_centering(setup):
    model, _, tokens = setup
    centered = read_point_covariance(model, tokens, center=True)[0]
    raw = read_point_covariance(model, tokens, center=False)[0]
    assert not torch.allclose(centered, raw)
    assert torch.linalg.eigvalsh(raw).min() > -1e-9      # still PSD


# -- the two degenerate merges --------------------------------------------


def test_averaging_readers_breaks_the_merge_with_a_zero_model(setup):
    """The failure that hid behind the capacity story: nothing was added, yet it broke."""
    model, _, tokens = setup
    merged = _sum_models([model.clone(), zeroed(model)], reader_rule="mean")
    assert _logit_gap(merged, model, tokens) > 1e-2


def test_summing_readers_breaks_the_merge_with_a_duplicate(setup):
    model, _, tokens = setup
    merged = _sum_models([model.clone(), model.clone()], reader_rule="sum")
    assert _logit_gap(merged, model, tokens) > 1e-2


def test_wiener_readers_are_exact_against_a_zero_model(setup):
    model, _, tokens = setup
    merged = _wiener_readers([model.clone(), zeroed(model)], [tokens, tokens])
    assert _logit_gap(merged, model, tokens) < 1e-6


def test_wiener_readers_are_exact_against_a_duplicate(setup):
    model, _, tokens = setup
    merged = _wiener_readers([model.clone(), model.clone()], [tokens, tokens])
    # looser than the zero case: the pooled second moment of N copies is
    # rank-deficient, so the pseudo-inverse drops a little genuine signal
    assert _logit_gap(merged, model, tokens) < 1e-3


def test_wiener_readers_hold_up_for_larger_n(setup):
    model, _, tokens = setup
    for n in (3, 5):
        models = [model.clone() for _ in range(n)]
        merged = _wiener_readers(models, [tokens] * n)
        assert _logit_gap(merged, model, tokens) < 1e-3, n


def test_wiener_is_insensitive_to_the_rank_tolerance(setup):
    """A result that only holds at one epsilon is a tuning artifact, not a rule.

    ``eps`` is a relative eigenvalue tolerance for the pseudo-inverse. Sweeping
    it over ten orders of magnitude must not move the answer -- which is what
    distinguishes a numerical cutoff from the modeling knobs that made the
    original capacity law untestable.
    """
    model, _, tokens = setup
    gaps = [
        _logit_gap(
            _wiener_readers([model.clone(), zeroed(model)], [tokens, tokens], eps=eps),
            model, tokens,
        )
        for eps in (1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
    ]
    assert max(gaps) < 1e-6, gaps
    assert max(gaps) - min(gaps) < 1e-9, gaps


def test_centering_the_projector_destroys_the_model(setup):
    """Pins the bug: a centered projector is blind along the mean direction.

    The residual stream's mean is functionally load-bearing, so a projector
    built from centered covariances zeroes the readers exactly where they carry
    signal. This regression test exists because that failure looked, from the
    outside, exactly like a capacity limit.
    """
    import fusion.capacity.merge as merge_mod

    model, _, tokens = setup
    real = merge_mod.read_point_covariance

    def centered(m, t, center=False):
        return real(m, t, center=True)

    merge_mod.read_point_covariance = centered
    try:
        broken = _wiener_readers([model.clone(), zeroed(model)], [tokens, tokens])
    finally:
        merge_mod.read_point_covariance = real
    assert _logit_gap(broken, model, tokens) > 1e-2
