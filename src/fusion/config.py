"""Configuration objects shared across the testbed."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of a specialist.

    The defaults are chosen to maximize the exploitable symmetry group:
    gainless RMSNorm, no biases, ReLU (positively homogeneous, so the MLP
    scaling symmetry is exact). See ``docs/DESIGN.md``.
    """

    vocab_size: int
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int | None = None
    max_seq_len: int = 16
    activation: str = "relu"
    tie_embeddings: bool = False

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        return self.d_model // self.n_heads

    @property
    def ff_dim(self) -> int:
        return self.d_ff if self.d_ff is not None else 4 * self.d_model

    def with_(self, **kw) -> ModelConfig:
        return replace(self, **kw)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1.0
    warmup: int = 100
    seed: int = 0
    eval_every: int = 500
    eval_batches: int = 8
    device: str = "auto"
    log: bool = True


@dataclass(frozen=True)
class FusionConfig:
    """Knobs for the fusion pipeline itself."""

    # Phase 1
    procrustes_iters: int = 30
    procrustes_tol: float = 1e-8
    canonicalize_heads: bool = True
    canonicalize_mlp: bool = True

    # Phase 2
    # "full" carries the whole spectrum into an energy-weighted overlap objective
    # and applies no cutoff at all; "participation" and "threshold" are the two
    # cutoff measures, kept for comparison (see docs/FINDINGS.md).
    rank_measure: str = "full"
    rank_threshold: float = 0.99          # energy fraction, used only by measure="threshold"
    stiefel_steps: int = 300
    stiefel_lr: float = 0.05
    disentangle: bool = True

    # Phase 3
    repair: bool = True
    repair_batches: int = 16

    calib_batch_size: int = 512
    extras: dict = field(default_factory=dict)
