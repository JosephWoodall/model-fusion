"""The Phase 1 pipeline: map each model independently into the canonical frame.

Order matters, and it is forced by the group structure (``docs/DESIGN.md``):

1. **Residual ``O(d)``** — fixed first, from the shared-vocabulary embeddings.
   Everything downstream is expressed relative to this frame.
2. **Head ``GL(d_head)``** — internal to each head, so it commutes with (1) and
   can be canonicalized afterwards without disturbing the residual frame.
3. **Head permutation, MLP permutation and scaling** — the discrete leftovers.

Steps 1-2 are reference-free.  Step 3's permutations are only *determined*
relative to something, so the pipeline first applies reference-free orderings
(by energy) and then, optionally, refines them against the consensus model.
That refinement is the one place an ordering could sneak in, so it is
explicitly opt-in via ``refine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..config import FusionConfig
from ..model import Transformer, require_same_arch
from ..symmetry import apply_head_permutation
from .complement import anchor_basis, complement_conditioning, complete_residual_frame_
from .heads import align_heads_to, canonical_head_order, canonicalize_heads
from .mlp import align_mlp_to, canonicalize_mlp
from .procrustes import align_residual_, frame_disagreement


@dataclass
class CanonicalizationReport:
    """What the alignment actually did, for attributing later failures."""

    rotations: list[Tensor] = field(default_factory=list)
    embedding_disagreement_before: float = float("nan")
    embedding_disagreement_after: float = float("nan")
    refined: bool = False
    complement_dim: int = 0
    complement_energy: float = float("nan")
    complement_min_gap: float = float("nan")
    complement_fixed: bool = False

    def improvement(self) -> float:
        before, after = self.embedding_disagreement_before, self.embedding_disagreement_after
        return float("nan") if before == 0 else 1.0 - after / before

    def __str__(self) -> str:
        head = (
            f"embedding disagreement {self.embedding_disagreement_before:.4e} -> "
            f"{self.embedding_disagreement_after:.4e} "
            f"({100 * self.improvement():.1f}% reduction)"
        )
        if not self.complement_fixed:
            return head + "; complement frame NOT fixed (no calibration data)"
        return (
            f"{head}; complement dim {self.complement_dim} carrying "
            f"{100 * self.complement_energy:.1f}% of residual energy, "
            f"min eigen-gap {self.complement_min_gap:.3f}"
        )


@torch.no_grad()
def canonicalize(
    model: Transformer,
    cfg: FusionConfig | None = None,
) -> Transformer:
    """Canonicalize the internal (per-head, per-MLP) gauge of a single model, in place.

    This half of canonicalization needs no other model and no data, so it is
    genuinely per-model.  The residual frame needs the consensus and is handled
    by :func:`canonicalize_all`.
    """
    cfg = cfg or FusionConfig()
    if cfg.canonicalize_heads:
        canonicalize_heads(model)
        for li in range(model.cfg.n_layers):
            apply_head_permutation(model, li, canonical_head_order(model, li))
    if cfg.canonicalize_mlp:
        canonicalize_mlp(model)
    return model


@torch.no_grad()
def canonicalize_all(
    models: list[Transformer],
    cfg: FusionConfig | None = None,
    calib_tokens: Tensor | None = None,
    refine: bool = True,
    inplace: bool = False,
) -> tuple[list[Transformer], CanonicalizationReport]:
    """Bring N models into one canonical frame.

    Returns the canonicalized models and a report.  With ``inplace=False``
    (the default) the inputs are left untouched, because the specialists are
    also the ceiling reference and must not be silently mutated.
    """
    cfg = cfg or FusionConfig()
    require_same_arch(*models)
    work = models if inplace else [m.clone() for m in models]

    report = CanonicalizationReport(
        embedding_disagreement_before=frame_disagreement(work),
    )

    # 1a. residual O(d) on the anchor span, from the shared vocabulary
    report.rotations = align_residual_(work, iters=cfg.procrustes_iters, tol=cfg.procrustes_tol)
    report.embedding_disagreement_after = frame_disagreement(work)

    # 1b. the anchor spans at most V + T directions.  Whatever it does not span,
    #     Procrustes leaves arbitrary -- and attention and MLP outputs write
    #     there.  Pin that subspace from the activation covariance.
    if calib_tokens is not None:
        complete_residual_frame_(work, calib_tokens)
        basis = anchor_basis(work)
        cond = complement_conditioning(work[0], _first(calib_tokens).to(work[0].embed.device),
                                       basis.to(work[0].embed))
        report.complement_dim = cond["complement_dim"]
        report.complement_energy = cond["energy_fraction"]
        report.complement_min_gap = cond["min_relative_gap"]
        report.complement_fixed = True

    # 2-3. internal gauge, per model, reference-free
    for m in work:
        canonicalize(m, cfg)

    # 3b. optional refinement of the discrete leftovers against model 0
    if refine and len(work) > 1:
        for m in work[1:]:
            if cfg.canonicalize_heads:
                align_heads_to(work[0], m)
            if cfg.canonicalize_mlp:
                align_mlp_to(work[0], m, _first(calib_tokens))
        report.refined = True

    return work, report


def _first(calib_tokens: Tensor | list[Tensor] | None) -> Tensor | None:
    """One shared batch, for the steps that compare two models on identical input."""
    if calib_tokens is None or isinstance(calib_tokens, Tensor):
        return calib_tokens
    return calib_tokens[0]
