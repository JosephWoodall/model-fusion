"""Reference points every merge result is reported against.

The floor (naive average) and the ceiling (each specialist on its own task)
live in :mod:`fusion.capacity.merge` and :mod:`fusion.train`. This package holds
the three competitors that actually matter:

* :func:`router_accuracy` — keep all N models, dispatch by task. The ``N*P``
  parameter bound; a merge that does not beat it on parameters has no reason to
  exist.
* :func:`distill` — train one student on the N teachers. The honest strong
  competitor. **Merging will not beat distillation on accuracy.** It wins on
  being data-free and O(seconds), and that is the comparison to state, not omit.
* joint training — in :func:`fusion.train.train_joint`.
"""

from .distill import distill
from .router import Router, router_accuracy

__all__ = ["distill", "Router", "router_accuracy"]
