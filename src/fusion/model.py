"""A bias-free, gainless-RMSNorm decoder-only transformer.

Every weight is stored so that the residual dimension is the *leading* axis of a
matrix that reads from the stream and the *trailing* axis of one that writes to
it.  That convention is what makes the residual rotation in
:mod:`fusion.symmetry` a uniform ``R.T @ W`` / ``W @ R`` rewrite.

The RMSNorm layers carry no learned gain.  A gain vector exists only as an
optional parameter used by Phase 3 repair; :meth:`Transformer.fold_gains`
absorbs it back into the following read-matrix so that the model returns to the
gainless form the orthogonal-invariance argument requires.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig

ACTIVATIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
}

#: Activations under which the MLP *scaling* symmetry is exact.
HOMOGENEOUS_ACTIVATIONS = frozenset({"relu"})


class RMSNorm(nn.Module):
    """RMSNorm with an optional gain.

    With ``gain is None`` (the trained state) the layer is exactly equivariant
    to orthogonal transforms of its input: ``norm(x @ R) == norm(x) @ R``.
    Phase 3 attaches a gain, refits it, and then folds it away again.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gain: nn.Parameter | None = None

    def enable_gain(self, device=None, dtype=None) -> nn.Parameter:
        if self.gain is None:
            self.gain = nn.Parameter(torch.ones(self.dim, device=device, dtype=dtype))
        return self.gain

    def disable_gain(self) -> Tensor | None:
        """Detach and return the gain, leaving the layer gainless."""
        if self.gain is None:
            return None
        g = self.gain.detach().clone()
        self.gain = None
        return g

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = x * rms
        return out if self.gain is None else out * self.gain

    def extra_repr(self) -> str:
        return f"dim={self.dim}, gain={'yes' if self.gain is not None else 'no'}"


class Attention(nn.Module):
    """Multi-head attention, causal, no biases.

    Weights are exposed as plain parameters rather than ``nn.Linear`` modules
    because the alignment code rewrites them directly and the transpose
    conventions of ``nn.Linear`` only add confusion.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        d, dh = cfg.d_model, cfg.n_heads * cfg.d_head
        self.w_q = nn.Parameter(torch.empty(d, dh))
        self.w_k = nn.Parameter(torch.empty(d, dh))
        self.w_v = nn.Parameter(torch.empty(d, dh))
        self.w_o = nn.Parameter(torch.empty(dh, d))
        self._init()

    def _init(self) -> None:
        d = self.w_q.shape[0]
        std = d**-0.5
        for p in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.normal_(p, std=std)

    def heads(self, name: str) -> Tensor:
        """Return a weight reshaped to ``[n_heads, ...]`` head-major blocks.

        ``q``/``k``/``v`` come back as ``[n_heads, d_model, d_head]`` and ``o``
        as ``[n_heads, d_head, d_model]``.
        """
        if name == "o":
            return self.w_o.view(self.n_heads, self.d_head, -1)
        w = {"q": self.w_q, "k": self.w_k, "v": self.w_v}[name]
        d = w.shape[0]
        return w.view(d, self.n_heads, self.d_head).permute(1, 0, 2)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q = (x @ self.w_q).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = (x @ self.w_k).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = (x @ self.w_v).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        z = z.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return z @ self.w_o


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {cfg.activation!r}")
        self.activation = cfg.activation
        self.act = ACTIVATIONS[cfg.activation]
        self.w_in = nn.Parameter(torch.empty(cfg.d_model, cfg.ff_dim))
        self.w_out = nn.Parameter(torch.empty(cfg.ff_dim, cfg.d_model))
        nn.init.normal_(self.w_in, std=cfg.d_model**-0.5)
        nn.init.normal_(self.w_out, std=cfg.ff_dim**-0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x @ self.w_in) @ self.w_out


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_attn = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln_mlp = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_attn(x))
        return x + self.mlp(self.ln_mlp(x))


class Transformer(nn.Module):
    """The specialist architecture.

    All N models in an experiment share this config and this vocabulary; only
    seed and training data differ.  Merging weights across differing
    architectures is meaningless, so the config is carried on the module and
    compared on every fusion call.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
        self.pos = nn.Parameter(torch.empty(cfg.max_seq_len, cfg.d_model))
        nn.init.normal_(self.embed, std=cfg.d_model**-0.5)
        nn.init.normal_(self.pos, std=cfg.d_model**-0.5)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = RMSNorm(cfg.d_model)
        self.unembed: nn.Parameter | None
        if cfg.tie_embeddings:
            self.unembed = None
        else:
            self.unembed = nn.Parameter(torch.empty(cfg.d_model, cfg.vocab_size))
            nn.init.normal_(self.unembed, std=cfg.d_model**-0.5)

    # -- structure ---------------------------------------------------------

    @property
    def unembed_matrix(self) -> Tensor:
        """``[d, V]``, whether or not embeddings are tied."""
        return self.embed.t() if self.unembed is None else self.unembed

    def norms(self) -> list[RMSNorm]:
        out: list[RMSNorm] = []
        for b in self.blocks:
            out += [b.ln_attn, b.ln_mlp]
        out.append(self.ln_f)
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- forward -----------------------------------------------------------

    def forward(self, tokens: Tensor) -> Tensor:
        T = tokens.shape[1]
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.embed[tokens] + self.pos[:T]
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x) @ self.unembed_matrix

    def loss(self, tokens: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
        logits = self(tokens)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )

    # -- repair support ----------------------------------------------------

    def enable_gains(self) -> list[nn.Parameter]:
        """Attach a learnable gain to every RMSNorm (Phase 3), on the model's device."""
        ref = next(self.parameters())
        return [n.enable_gain(device=ref.device, dtype=ref.dtype) for n in self.norms()]

    def fold_gains(self) -> None:
        """Absorb every RMSNorm gain into the matrices that read from it.

        Restores the gainless form required for orthogonal invariance.  This is
        exact: ``(x * g) @ W == x @ (diag(g) @ W)``.
        """
        for block in self.blocks:
            g = block.ln_attn.disable_gain()
            if g is not None:
                for name in ("w_q", "w_k", "w_v"):
                    w = getattr(block.attn, name)
                    w.data = g[:, None] * w.data
            g = block.ln_mlp.disable_gain()
            if g is not None:
                block.mlp.w_in.data = g[:, None] * block.mlp.w_in.data
        g = self.ln_f.disable_gain()
        if g is not None:
            if self.unembed is None:
                raise RuntimeError(
                    "cannot fold the final gain into tied embeddings; "
                    "train with tie_embeddings=False when using repair"
                )
            self.unembed.data = g[:, None] * self.unembed.data

    # -- convenience -------------------------------------------------------

    def clone(self) -> Transformer:
        """A deep copy preserving device, dtype, and any attached gains."""
        ref = next(self.parameters())
        out = Transformer(self.cfg).to(device=ref.device, dtype=ref.dtype)
        for norm, src in zip(out.norms(), self.norms(), strict=True):
            if src.gain is not None:
                norm.enable_gain(device=ref.device, dtype=ref.dtype)
        out.load_state_dict(self.state_dict())
        return out.to(device=ref.device, dtype=ref.dtype)

    @torch.no_grad()
    def interpolate_(self, other: Transformer, alpha: float) -> Transformer:
        """In-place ``(1-alpha)*self + alpha*other``; returns self."""
        require_same_arch(self, other)
        for p, q in zip(self.parameters(), other.parameters(), strict=True):
            p.mul_(1.0 - alpha).add_(q, alpha=alpha)
        return self

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"d_model={c.d_model}, n_layers={c.n_layers}, n_heads={c.n_heads}, "
            f"d_ff={c.ff_dim}, vocab={c.vocab_size}, params={self.num_params()}"
        )


def require_same_arch(*models: Transformer) -> ModelConfig:
    """Fail loudly when asked to merge models that are not comparable."""
    if not models:
        raise ValueError("no models given")
    cfg = models[0].cfg
    for m in models[1:]:
        if m.cfg != cfg:
            raise ValueError(f"architecture mismatch:\n  {cfg}\n  {m.cfg}")
    return cfg


def init_from_seed(cfg: ModelConfig, seed: int, device: str | torch.device = "cpu") -> Transformer:
    gen_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = Transformer(cfg)
    finally:
        torch.random.set_rng_state(gen_state)
    return model.to(device)


def positional_scale(cfg: ModelConfig) -> float:
    """The 1/sqrt(d_head) factor SDPA applies internally, exposed for circuit math."""
    return 1.0 / math.sqrt(cfg.d_head)
