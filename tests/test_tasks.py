"""The testbed itself: shared vocabulary, correct labels, and a usable loss mask."""

import pytest
import torch

from fusion.tasks import (
    IGNORE,
    OVERLAP_GROUPS,
    TASK_REGISTRY,
    build_vocab,
    make_tasks,
    task_family,
)


def test_all_registered_tasks_build_and_produce_correct_answers():
    names = list(TASK_REGISTRY)
    tasks = make_tasks(names, p=13)
    gen = torch.Generator().manual_seed(0)
    for task in tasks:
        toks, targets = task.batch(64, gen)
        assert toks.shape == (64, task.seq_len)
        assert targets.shape == toks.shape
        assert (targets != IGNORE).sum(dim=1).eq(task.answer_len).all()
        # the unmasked targets are exactly the answer block
        ans = toks[:, -task.answer_len:]
        assert torch.equal(targets[targets != IGNORE].reshape(-1, task.answer_len), ans)


def test_modular_answers_are_actually_modular():
    tasks = {t.name: t for t in make_tasks(["mod_add", "mod_sub", "mod_mul"], p=17)}
    o = torch.tensor([[5, 9], [16, 3], [0, 0]])
    assert torch.equal(tasks["mod_add"].answer(o).squeeze(1), torch.tensor([14, 2, 0]))
    assert torch.equal(tasks["mod_sub"].answer(o).squeeze(1), torch.tensor([13, 13, 0]))
    assert torch.equal(tasks["mod_mul"].answer(o).squeeze(1), torch.tensor([11, 14, 0]))


def test_sequence_tasks():
    tasks = {t.name: t for t in make_tasks(["reverse", "sort", "copy"], p=10)}
    o = torch.tensor([[3, 1, 4, 1, 5]])
    assert torch.equal(tasks["reverse"].answer(o), torch.tensor([[5, 1, 4, 1, 3]]))
    assert torch.equal(tasks["sort"].answer(o), torch.tensor([[1, 1, 3, 4, 5]]))
    assert torch.equal(tasks["copy"].answer(o), o)


def test_all_tasks_in_a_family_share_one_vocabulary():
    """The anchor for Procrustes: token ids must mean the same thing in every model."""
    for overlap in OVERLAP_GROUPS:
        tasks = task_family(overlap, 5, p=11)
        assert all(t.vocab is tasks[0].vocab for t in tasks)
        ops = {t.name: t.vocab.op(t.name) for t in tasks}
        assert len(set(ops.values())) == len(set(ops))


def test_op_tokens_are_distinct_from_digits_and_controls():
    vocab = build_vocab(["mod_add", "mod_sub"], p=7)
    specials = {vocab.bos, vocab.ans, vocab.pad, vocab.op("mod_add"), vocab.op("mod_sub")}
    assert len(specials) == 5
    assert min(specials) >= vocab.p
    assert vocab.size == 7 + 3 + 2


def test_task_family_scales_past_the_number_of_task_types():
    tasks = task_family("high", 8, p=11)     # only 2 base types
    assert len(tasks) == 8
    assert len({t.name for t in tasks}) == 2


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unknown tasks"):
        build_vocab(["not_a_task"], p=5)
