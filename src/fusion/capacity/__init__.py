"""Phase 2 — capacity-aware fusion. The contribution.

Alignment says *where* the models' features live. Capacity says whether they
*fit*. The falsifiable claim: a merge succeeds iff ``sum_k r_k <= d``.
"""

from .merge import CapacityReport, capacity_ratio, fuse, naive_average
from .rank import RankProfile, activation_covariance, effective_rank, used_subspace
from .stiefel import disentangle, subspace_overlap

__all__ = [
    "RankProfile",
    "activation_covariance",
    "effective_rank",
    "used_subspace",
    "disentangle",
    "subspace_overlap",
    "fuse",
    "naive_average",
    "capacity_ratio",
    "CapacityReport",
]
