"""The N*P bound: keep every specialist and dispatch to the right one.

This is trivially the best achievable accuracy — it *is* the specialists — at N
times the parameters. Its role is to make the merge's parameter saving explicit.
A merged model that matches the router's accuracy at 1/N the parameters is the
ideal outcome; anything below it is the price of fusion, and quoting the price
requires quoting this number.

The router here uses the **oracle** task label carried by the op token. A
learned router would be a confound: its errors would mix into the accuracy and
make the reference point worse than the thing it is bounding.
"""

from __future__ import annotations

import torch

from ..model import Transformer
from ..tasks import Task, accuracy


class Router:
    """Oracle dispatch from task name to specialist."""

    def __init__(self, models: dict[str, Transformer]):
        self.models = models

    def for_task(self, task: Task) -> Transformer:
        if task.name not in self.models:
            raise KeyError(f"no specialist for task {task.name!r}")
        return self.models[task.name]

    def num_params(self) -> int:
        return sum(m.num_params() for m in self.models.values())

    @torch.no_grad()
    def __call__(self, tokens: torch.Tensor, task: Task) -> torch.Tensor:
        return self.for_task(task)(tokens)


def router_accuracy(router: Router, tasks: list[Task], n: int = 2048) -> dict[str, float]:
    out = {}
    for task in tasks:
        model = router.for_task(task)
        out[task.name] = accuracy(model, task, n=n, device=next(model.parameters()).device)
    return out
