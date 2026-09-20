# Findings

Running log of results from this testbed. Dates are when the run was made. Every
claim here is reproducible with the command given.

---

## 2026-09-20 — Phase 0/1 go/no-go

`python -m experiments.exp0_sanity --steps 1500 --d 128 --p 47`

Testbed: `d=128`, 2 layers, 4 heads, `p=47`, modular addition, two seeds.

### The embedding anchor is rank-deficient, and that is the whole problem

"Orthogonal Procrustes on the embeddings" determines the residual frame **only on
the span of the anchor matrices**. Those matrices have `V + T` rows. Here that is
51 + 6 = 57 directions out of `d = 128`. On the remaining 71, Procrustes returns
an *arbitrary* rotation — the objective is flat there.

Those directions are not inert. Measured energy of the residual stream:

| Layer | Inside anchor span | Outside |
|-------|-------------------|---------|
| 0 | 100% | 0% |
| 1 | 45.4% | **54.6%** |
| 2 | 45.3% | **54.7%** |

Layer 0 is inside by construction — it is exactly `E + P`. But attention and MLP
outputs write wherever they like, and by layer 1 the majority of the
computation has moved into the subspace the anchor cannot see.

**Symptom:** embedding disagreement falls to `1e-17` while the interpolation
barrier does not move at all. Recovery error against a known planted rotation:
`0.415`. This is easy to misread as "the models learned different solutions".

**Fix, in `fusion.align.complement`,** in two parts:

1. *Augment the anchor.* The unembedding `U` is also expressed in the shared
   token basis, one row of `Uᵀ` per output token. Stacking `[E; P; Uᵀ]` raises
   the anchor rank from 57 to 108 and cuts the undetermined subspace from 71
   dimensions carrying 55% of the energy to 20 carrying 8.4%.
2. *Pin the complement from the data.* On what is left there is no shared basis
   to match against, so each model is put into its own canonical frame there:
   diagonalize the residual activation covariance restricted to the complement,
   order eigenvectors by eigenvalue, fix signs by the skewness of the projected
   activations. Reference-free, so canonicalization stays per-model and
   order-invariant over N.

### Control A — pure frame difference

A trained model against an exactly rotated copy of itself.

| | before | after |
|---|---|---|
| interpolation barrier | 4.74 | **0.000000** |
| embedding disagreement | 9.5e-4 | 3.9e-17 |
| worst parameter difference | — | 6.2e-6 (float32 noise) |

The canonicalizer recovers a planted rotation exactly. The machinery works.

### Control B — same task, different seeds

| | before | after |
|---|---|---|
| interpolation barrier | 4.69 | 6.57 |
| layerwise CKA | 0.948 / 0.886 / 0.816 | unchanged (CKA is rotation-invariant) |
| merged accuracy | 0.025 | 0.037 (chance = 0.021) |

**The barrier does not collapse.** With Control A passing, this is not an
alignment failure. The mechanistic readout says what it is:

```
mod_add-s0: key freqs [1, 7, 11, 18, 23]
mod_add-s1: key freqs [1, 13, 16, 22]
```

The two models grokked **different Fourier frequencies**. They compute the same
function through different circuits, and no element of `O(d)` maps one onto the
other. CKA agrees independently, and CKA cannot be moved by alignment.

**This is a capacity result, not an alignment result.** Two different circuits
need two sets of directions, and whether they fit is exactly what `Σr_k ≤ d`
decides.

### What this means for the program

The original plan's go/no-go was: *"if the interpolation barrier doesn't collapse
[on same-task/different-seed pairs], stop and fix alignment before touching
anything else."* Run as written, that test would have sent us to fix an aligner
that is provably exact. **Control A is what distinguishes the two cases, and it
belongs in the gate.** The gate is now:

- Control A fails → the aligner is broken. Fix it.
- Control A passes, Control B fails → solution diversity. A Phase 2 question.
- Both pass → proceed.

---

## 2026-09-20 — Phase 1 against the baselines

`python -m experiments.exp1_alignment_baselines --steps 800 --d 128`

Merged accuracy after each alignment method, then averaging. Chance is 0.021.

| Method | Known frame difference | Independent seeds |
|--------|-----------------------|-------------------|
| naive average (floor) | 0.053 | 0.025 |
| Git Re-Basin (weight matching) | 0.022 | 0.020 |
| activation matching | 0.050 | 0.023 |
| OT fusion (Sinkhorn) | 0.035 | 0.023 |
| ZipIt! feature merging | 0.044 | 0.023 |
| **canonicalization (this repo)** | **1.000** | 0.024 |

Left column: every method is handed a difference it can in principle undo. Only
the one that searches `O(d)` undoes it — barrier `4.68 → 0.0000`. Every
permutation-based method stays at chance, because the difference is not a
permutation. This is the concrete value of the from-scratch, gainless-RMSNorm,
bias-free architecture: it is what makes the larger group available.

Right column: no method recovers anything, for the reason in Control B above.
Reported rather than omitted.

---

## Open questions

- Does the knee in degradation actually land at `Σr_k/d = 1.0`? (`exp2_capacity`)
- Solution diversity is the binding constraint at `d=128`. Does it shrink at
  larger `d`, or is it intrinsic to grokking modular arithmetic?
- Does training the N specialists from a **shared initialization** remove the
  frequency divergence, and if so, is a merge that requires shared init still
  interesting?
- `diagnose_merge` reports "superposed" on merges whose accuracy is at chance:
  the embedding Fourier structure survives but the circuit consuming it does
  not. A downstream-circuit diagnostic is needed to close that gap.
