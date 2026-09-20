"""Phase 2 — capacity-aware fusion. The contribution.

Alignment says *where* the models' features live. Capacity says whether they
*fit*. The falsifiable claim, restated without a rank cutoff: a merge succeeds
while the energy-weighted interference ratio stays below 1.0 (0 dB).
"""

from .functional import (
    FunctionalRank,
    capacity_verdict,
    energy_basis,
    functional_rank,
    restrict_to,
)
from .interference import InterferenceReport, measure_interference, signal_retention
from .merge import CapacityReport, capacity_ratio, fuse, naive_average
from .rank import RankProfile, activation_covariance, effective_rank, used_subspace
from .stiefel import (
    KNEE,
    disentangle,
    interference_ratio,
    rearrangement_floor,
    subspace_overlap,
)

__all__ = [
    "RankProfile",
    "activation_covariance",
    "effective_rank",
    "used_subspace",
    "disentangle",
    "subspace_overlap",
    "interference_ratio",
    "rearrangement_floor",
    "KNEE",
    "measure_interference",
    "signal_retention",
    "InterferenceReport",
    "functional_rank",
    "FunctionalRank",
    "capacity_verdict",
    "energy_basis",
    "restrict_to",
    "fuse",
    "naive_average",
    "capacity_ratio",
    "CapacityReport",
]
