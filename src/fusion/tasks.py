"""Task suite with a relatedness dial, over a single shared vocabulary.

Every task in a run uses the *same* tokenizer and the *same* vocabulary.  That
is the anchor for Phase 1: because all models' embedding matrices are expressed
in a common token basis, orthogonal Procrustes on the embeddings is meaningful.

Sequence format is fixed and identical across tasks::

    [BOS] a b [OP] [ANS] y0 y1 ... y_{L-1}

Only the answer positions carry a loss; everything else is ``IGNORE``.  Binary
tasks emit a single answer token; sequence tasks (reverse, sort, copy) emit a
whole block.  Keeping one format across the suite means the merged model never
has to reconcile different input conventions, which would confound the
capacity question with a tokenization question.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

IGNORE = -100

#: Relatedness dial.  These groupings are the independent variable of Phase 4.
OVERLAP_GROUPS: dict[str, tuple[str, ...]] = {
    "high": ("mod_add", "mod_sub"),
    "medium": ("mod_mul", "parity"),
    "disjoint": ("reverse", "sort", "copy"),
}


@dataclass(frozen=True)
class Vocab:
    """Shared vocabulary: digits 0..p-1, then the control tokens, then one op token per task.

    Built once per experiment from the union of all tasks in play, so that
    every model sees exactly the same token ids.
    """

    p: int
    tasks: tuple[str, ...]

    @property
    def bos(self) -> int:
        return self.p

    @property
    def ans(self) -> int:
        return self.p + 1

    @property
    def pad(self) -> int:
        return self.p + 2

    def op(self, task: str) -> int:
        return self.p + 3 + self.tasks.index(task)

    @property
    def size(self) -> int:
        return self.p + 3 + len(self.tasks)

    def decode(self, tok: int) -> str:
        if tok < self.p:
            return str(tok)
        return {self.bos: "<bos>", self.ans: "<ans>", self.pad: "<pad>"}.get(
            tok, f"<{self.tasks[tok - self.p - 3]}>"
        )


class Task:
    """Base class. A task is a deterministic map from operands to answer tokens."""

    name: str = ""
    n_operands: int = 2
    answer_len: int = 1

    def __init__(self, vocab: Vocab):
        if self.name not in vocab.tasks:
            raise ValueError(f"task {self.name!r} not in vocab tasks {vocab.tasks}")
        self.vocab = vocab
        self.p = vocab.p

    # -- to implement ------------------------------------------------------

    def answer(self, operands: Tensor) -> Tensor:
        """``[B, n_operands] -> [B, answer_len]``."""
        raise NotImplementedError

    def sample_operands(self, n: int, generator: torch.Generator | None) -> Tensor:
        return torch.randint(0, self.p, (n, self.n_operands), generator=generator)

    # -- shared ------------------------------------------------------------

    @property
    def seq_len(self) -> int:
        # [BOS] operands... [OP] [ANS] answer...
        return 1 + self.n_operands + 2 + self.answer_len

    def batch(
        self, n: int, generator: torch.Generator | None = None, device=None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(tokens, targets)``, both ``[n, seq_len]``.

        ``targets`` is next-token shifted and masked to the answer positions, so
        the loss measures exactly the task and nothing else.
        """
        v = self.vocab
        ops = self.sample_operands(n, generator)
        ans = self.answer(ops)
        prefix = torch.full((n, 1), v.bos)
        toks = torch.cat(
            [prefix, ops, torch.full((n, 1), v.op(self.name)), torch.full((n, 1), v.ans), ans],
            dim=1,
        )
        targets = torch.full_like(toks, IGNORE)
        # position i predicts token i+1; answers start at index 1 + n_operands + 2
        start = 1 + self.n_operands + 2
        targets[:, start - 1 : -1] = toks[:, start:]
        if device is not None:
            toks, targets = toks.to(device), targets.to(device)
        return toks, targets

    def answer_positions(self) -> slice:
        start = 1 + self.n_operands + 2
        return slice(start - 1, start - 1 + self.answer_len)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p}, len={self.seq_len})"


# --------------------------------------------------------------------------
# high overlap: the same Fourier circuit, different phase
# --------------------------------------------------------------------------


class ModAdd(Task):
    name = "mod_add"

    def answer(self, o: Tensor) -> Tensor:
        return ((o[:, 0] + o[:, 1]) % self.p)[:, None]


class ModSub(Task):
    name = "mod_sub"

    def answer(self, o: Tensor) -> Tensor:
        return ((o[:, 0] - o[:, 1]) % self.p)[:, None]


# --------------------------------------------------------------------------
# medium overlap
# --------------------------------------------------------------------------


class ModMul(Task):
    name = "mod_mul"

    def answer(self, o: Tensor) -> Tensor:
        return ((o[:, 0] * o[:, 1]) % self.p)[:, None]


class Parity(Task):
    """Parity of the count of odd operands, over a wider operand block."""

    name = "parity"
    n_operands = 6

    def sample_operands(self, n: int, generator: torch.Generator | None) -> Tensor:
        return torch.randint(0, 2, (n, self.n_operands), generator=generator)

    def answer(self, o: Tensor) -> Tensor:
        return (o.sum(dim=1) % 2)[:, None]


# --------------------------------------------------------------------------
# disjoint: no arithmetic structure at all
# --------------------------------------------------------------------------


class Reverse(Task):
    name = "reverse"
    n_operands = 5
    answer_len = 5

    def answer(self, o: Tensor) -> Tensor:
        return o.flip(dims=(1,))


class Sort(Task):
    name = "sort"
    n_operands = 5
    answer_len = 5

    def answer(self, o: Tensor) -> Tensor:
        return o.sort(dim=1).values


class Copy(Task):
    """Induction/copy: emit the block that followed the first occurrence of the cue."""

    name = "copy"
    n_operands = 5
    answer_len = 5

    def answer(self, o: Tensor) -> Tensor:
        return o


TASK_REGISTRY: dict[str, type[Task]] = {
    cls.name: cls for cls in (ModAdd, ModSub, ModMul, Parity, Reverse, Sort, Copy)
}


def build_vocab(task_names: list[str] | tuple[str, ...], p: int = 47) -> Vocab:
    """One vocabulary covering every task in the run. Shared by all N models."""
    unknown = set(task_names) - set(TASK_REGISTRY)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    return Vocab(p=p, tasks=tuple(task_names))


def make_tasks(task_names: list[str] | tuple[str, ...], p: int = 47) -> list[Task]:
    vocab = build_vocab(task_names, p)
    return [TASK_REGISTRY[name](vocab) for name in task_names]


def task_family(overlap: str, n: int, p: int = 47) -> list[Task]:
    """``n`` tasks at the requested relatedness level.

    When ``n`` exceeds the number of distinct task types at that overlap level,
    the extra models are *different data draws of the same task types* — which
    is the right way to scale N while holding relatedness fixed.  The caller
    gets distinct :class:`Task` instances sharing one vocabulary.
    """
    if overlap not in OVERLAP_GROUPS:
        raise ValueError(f"overlap must be one of {sorted(OVERLAP_GROUPS)}")
    base = OVERLAP_GROUPS[overlap]
    names = [base[i % len(base)] for i in range(n)]
    vocab = build_vocab(tuple(dict.fromkeys(names)), p)
    return [TASK_REGISTRY[name](vocab) for name in names]


def max_seq_len(tasks: list[Task]) -> int:
    return max(t.seq_len for t in tasks)


@torch.no_grad()
def accuracy(model, task: Task, n: int = 2048, generator=None, device=None) -> float:
    """Exact-match accuracy over the whole answer block.

    Exact match, not per-token: a partially correct sort is not a correct sort,
    and per-token accuracy would hide exactly the collapse we are looking for.
    """
    model.eval()
    toks, targets = task.batch(n, generator, device=device or next(model.parameters()).device)
    logits = model(toks)
    pred = logits.argmax(dim=-1)
    mask = targets != IGNORE
    correct = ((pred == targets) | ~mask).all(dim=1)
    return correct.float().mean().item()
