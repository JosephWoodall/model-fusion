"""Phase 4 — the diagnostics, and the grid that uses them."""

from .metrics import (
    cka,
    interpolation_barrier,
    layerwise_cka,
    normalized_accuracy,
    worst_task_accuracy,
)

__all__ = [
    "interpolation_barrier",
    "cka",
    "layerwise_cka",
    "worst_task_accuracy",
    "normalized_accuracy",
]
