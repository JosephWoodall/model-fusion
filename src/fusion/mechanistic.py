"""Phase 5 — read the circuits directly, and say *what* the merge did.

At this scale, on modular arithmetic, the circuits are legible. A network that
has learned modular addition represents each operand as a set of Fourier
components: the embedding of token ``a`` contains directions proportional to
``cos(2*pi*k*a/p)`` and ``sin(2*pi*k*a/p)`` for a handful of key frequencies
``k``, and the network combines them with trig identities to produce
``a + b mod p`` (Nanda et al.).

That makes a specific, checkable claim available after every merge. When a
merged model loses accuracy on modular subtraction, the question "why" has three
concrete answers, and :func:`diagnose_merge` distinguishes them:

* **overwritten** — the add model's frequencies survive, the sub model's are gone
* **superposed** — both sets of frequencies are present, with reduced amplitude
* **destroyed** — neither is present

Nobody validates merging this way, because at production scale you cannot. It
turns "it degraded 4%" into "here is the mechanism."
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import Transformer


def fourier_basis(p: int, dtype=torch.float64, device=None) -> tuple[Tensor, list[str]]:
    """Real Fourier basis over ``Z_p``: ``[p, p]`` rows, orthonormal.

    Row 0 is the constant; then ``cos(2*pi*k*x/p)``, ``sin(2*pi*k*x/p)`` for
    ``k = 1 .. (p-1)//2``.
    """
    x = torch.arange(p, dtype=dtype, device=device)
    rows = [torch.ones(p, dtype=dtype, device=device)]
    names = ["const"]
    for k in range(1, p // 2 + 1):
        ang = 2 * torch.pi * k * x / p
        rows.append(torch.cos(ang))
        names.append(f"cos{k}")
        if not (p % 2 == 0 and k == p // 2):
            rows.append(torch.sin(ang))
            names.append(f"sin{k}")
    B = torch.stack(rows)[:p]
    return B / B.norm(dim=1, keepdim=True), names[:p]


@dataclass
class FourierProfile:
    """How much of a model's numeral embeddings sits in each Fourier frequency."""

    power: Tensor          # [n_freqs] energy per frequency, summed over cos/sin
    names: list[str]
    p: int

    def key_frequencies(self, threshold: float = 0.05) -> list[int]:
        """Frequencies carrying more than ``threshold`` of the total power.

        A grokked modular-arithmetic model typically shows 3-6 of these, sharply
        above a near-zero floor. A model that has memorized instead shows a flat
        spectrum, which is itself a useful signal.
        """
        total = self.power.sum().clamp_min(1e-30)
        frac = self.power / total
        return [int(i) for i in torch.where(frac > threshold)[0].tolist()]

    def concentration(self) -> float:
        """Fraction of power in the top 6 frequencies. High = clean circuit."""
        top = torch.topk(self.power, min(6, self.power.numel())).values.sum()
        return float(top / self.power.sum().clamp_min(1e-30))

    def __str__(self) -> str:
        keys = self.key_frequencies()
        return (
            f"key freqs {keys} (concentration {self.concentration():.2f} "
            f"of power in top 6 of {self.power.numel()})"
        )


@torch.no_grad()
def fourier_profile(model: Transformer, p: int) -> FourierProfile:
    """Fourier power spectrum of the numeral embeddings ``E[0:p]``.

    Rotation invariant: the power is computed per frequency by projecting the
    *rows* of ``E`` onto the Fourier basis, and the residual-stream rotation acts
    on the columns. So this is comparable across models without aligning them
    first — which matters, because the whole point is to check what survived a
    merge, not to re-align it.
    """
    E = model.embed.detach()[:p].double()
    B, names = fourier_basis(p, dtype=E.dtype, device=E.device)
    coeffs = B @ E                     # [p, d]
    power_per_row = coeffs.pow(2).sum(dim=1)        # [p]

    freqs, labels = [], []
    i = 0
    while i < len(names):
        name = names[i]
        if name == "const":
            freqs.append(power_per_row[i])
            labels.append("const")
            i += 1
            continue
        k = int(name[3:])
        pair = power_per_row[i]
        if i + 1 < len(names) and names[i + 1].startswith("sin") and int(names[i + 1][3:]) == k:
            pair = pair + power_per_row[i + 1]
            i += 2
        else:
            i += 1
        freqs.append(pair)
        labels.append(f"freq{k}")
    return FourierProfile(power=torch.stack(freqs), names=labels, p=p)


@dataclass
class MergeDiagnosis:
    """What happened to each specialist's circuit inside the merged model."""

    survival: dict[str, float]     # per model: fraction of its key-frequency power retained
    verdict: str
    merged_keys: list[int]
    specialist_keys: dict[str, list[int]]

    def __str__(self) -> str:
        detail = ", ".join(f"{k}: {v:.2f}" for k, v in self.survival.items())
        return f"{self.verdict} (retained power — {detail})"


@torch.no_grad()
def diagnose_merge(
    merged: Transformer,
    specialists: dict[str, Transformer],
    p: int,
    threshold: float = 0.05,
    survive_at: float = 0.5,
) -> MergeDiagnosis:
    """Classify a merge as superposed, overwritten, or destroyed.

    For each specialist, take the frequencies it relies on and measure what
    fraction of the merged model's numeral-embedding power sits in them,
    relative to what that specialist itself devotes to them. Then:

    * all specialists retain > ``survive_at``  -> **superposed**
    * some do, some do not                     -> **overwritten**
    * none do                                  -> **destroyed**

    **Caveat, and it matters.** This reads the *embedding* Fourier structure
    only. Surviving frequencies mean the merged model still represents the
    operands the way the specialists did; they do not mean the downstream
    circuit that consumes them survived. A merge can therefore come back
    "superposed" while accuracy is at chance -- which is itself a useful
    reading: the inputs are intact and the computation over them is what broke,
    so the failure is downstream of the embeddings.
    """
    mp = fourier_profile(merged, p)
    m_power = mp.power / mp.power.sum().clamp_min(1e-30)

    survival, keys = {}, {}
    for name, model in specialists.items():
        sp = fourier_profile(model, p)
        s_power = sp.power / sp.power.sum().clamp_min(1e-30)
        k = sp.key_frequencies(threshold)
        keys[name] = k
        if not k:
            survival[name] = float("nan")
            continue
        idx = torch.tensor(k, device=m_power.device)
        survival[name] = float(m_power[idx].sum() / s_power[idx].sum().clamp_min(1e-30))

    alive = [v for v in survival.values() if v == v and v > survive_at]
    scored = [v for v in survival.values() if v == v]
    if not scored:
        verdict = "indeterminate (no clean circuit in the specialists)"
    elif len(alive) == len(scored):
        verdict = "superposed"
    elif alive:
        verdict = "overwritten"
    else:
        verdict = "destroyed"

    return MergeDiagnosis(
        survival=survival, verdict=verdict,
        merged_keys=mp.key_frequencies(threshold), specialist_keys=keys,
    )


@torch.no_grad()
def neuron_frequency_map(model: Transformer, layer: int, p: int) -> Tensor:
    """Dominant Fourier frequency of each MLP neuron's input direction: ``[d_ff]``.

    In a grokked model the hidden units cluster onto the same handful of key
    frequencies the embeddings use. Watching that clustering survive or dissolve
    across a merge is the neuron-level version of :func:`diagnose_merge`.
    """
    E = model.embed.detach()[:p].double()
    B, _ = fourier_basis(p, dtype=E.dtype, device=E.device)
    w_in = model.blocks[layer].mlp.w_in.detach().double()
    pre = E @ w_in                                   # [p, d_ff] response to each numeral
    coeffs = (B @ pre).pow(2)         # [p, d_ff] power per basis row
    return coeffs.argmax(dim=0)
