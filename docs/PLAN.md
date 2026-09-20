# The program

Three separable problems. Conflating them is why most from-scratch fusion attempts fail
ambiguously.

1. **Alignment** — the models live in different frames. Fixed by exploiting the architecture's
   symmetry group.
2. **Capacity** — N sets of features must fit in one parameter budget. This is a packing problem,
   not an alignment problem.
3. **Repair** — merged activations have wrong statistics even when 1 and 2 succeed.

Each gets its own diagnostic so a failure identifies its own cause.

---

## Phase 0 — Testbed designed for symmetry

Because the models are trained from scratch, the architecture is chosen to **maximize the symmetry
group we can exploit**. This is the biggest lever available and it is unavailable to anyone merging
third-party checkpoints.

- **RMSNorm, not LayerNorm**, with the learned gain folded into the following linear layer. An
  RMSNorm network is computationally invariant under orthogonal transforms of the residual stream
  (SliceGPT). Alignment group: `O(d)` rather than the permutations `P`.
- **No biases.** They pin the frame and buy little at this scale.
- **Shared tokenizer and vocab across all N models.** The anchor for Phase 1.
- **Identical architecture, different seeds and data.** Width and depth must match; weight merging
  is otherwise meaningless.

### Task suite, with a relatedness dial

| Overlap | Tasks |
|---------|-------|
| high | modular addition, modular subtraction |
| medium | multiply mod p, parity |
| disjoint | sequence reversal, sorting, induction/copy |

### The critical control

Two models trained on the **same task with different seeds**. If alignment cannot merge those
near-losslessly, nothing downstream means anything. This control isolates alignment from capacity
and is the gate on the whole program.

---

## Phase 1 — Canonicalization

Do **not** do pairwise alignment to a reference model: it is order-dependent and O(N²). Map each
model *independently* into a canonical frame. N-way alignment is then free and order-invariant,
which is exactly what "works over any N" requires.

**Residual stream: `O(d)`.** All models share a vocabulary, so the embedding matrices
`E_k ∈ R^{V×d}` are expressed in a common token basis. Orthogonal Procrustes — find `R_k`
minimizing `‖E_k R_k − Ē‖_F`, iterating `Ē` as the consensus — is a cheap, principled global
alignment. Apply `R_k` to every matrix reading from or writing to the stream. This is a hundred
lines and already beats naive averaging by a wide margin; start here.

**Attention heads: `GL(d_head)`.** QK is invariant under `W_Q → W_Q A`, `W_K → W_K A^{−T}`; OV
under `W_V → W_V B`, `W_O → B^{−1} W_O`. Canonicalize each head by taking the SVD of the QK
circuit `W_Q W_Kᵀ` and the OV circuit `W_V W_O`, rotating to singular-vector coordinates. Heads
then differ only by a permutation across heads — solve with Hungarian on circuit similarity.

**MLP hidden units: permutation + positive scaling** (for homogeneous activations). Fix scale by
normalizing rows, then match by activation correlation.

**Diagnostics.** Linear-interpolation barrier between aligned models on a shared task (should
collapse toward zero); layerwise CKA after alignment.

**Baselines implemented here.** Git Re-Basin weight matching and activation matching, OT fusion
(Sinkhorn), ZipIt!-style feature matching. The orthogonal-plus-Procrustes approach should beat all
of them. If it does not, that is informative and gets reported.

---

## Phase 2 — Capacity-aware fusion (the contribution)

After alignment there is still residual `O(d)` freedom. Spend it making the models' *used*
directions mutually orthogonal.

### The original formulation, and why it was replaced

1. For each model `k` and layer, run that model's own task data and take the activation
   covariance. Its effective rank `r_k` is the number of eigenvalues above threshold.
2. Solve for orthogonal `R_k` minimizing `Σ_{j≠k} ‖(R_k U_k)ᵀ (R_j U_j)‖²_F` by Stiefel-manifold
   optimization.
3. If `Σ_k r_k ≤ d` the subspaces can be made disjoint and the merge is a sum; otherwise an
   output-space QP picks what to discard under the budget, reporting captured energy `ρ`.

This yields the condition **merge succeeds iff `Σ_k r_k ≤ d`** — which turned out to be
**untestable in both directions**. A thresholded `r_k` is set by its threshold (39 / 98 / 124 at
0.90 / 0.99 / 0.999 for one model at `d=128`), so the predicted knee moves with an arbitrary
constant. The threshold-free participation ratio is so generous that every configuration passes
while every merge still collapses. Neither bracket contains a transition. See
[`FINDINGS.md`](FINDINGS.md).

### The energy-weighted formulation (current)

Drop the cutoff and carry the whole spectrum. With `M_k = R_k U_k Λ_k^{1/2}` and
`A_k = M_k M_kᵀ = R_k C_k R_kᵀ`:

> minimize over orthogonal `R_1..R_N`:  `Σ_{j≠k} ‖M_kᵀ M_j‖²_F = Σ_{j≠k} tr(A_k A_j)`

the Frobenius inner product between rotated covariances. No threshold appears anywhere: a
direction contributes in proportion to the energy actually on it.

Three numbers come out of the same traces:

| Quantity | Definition | Role |
|---|---|---|
| weighted overlap | `Σ tr(A_kA_j) / Σ ‖A_k‖_F‖A_j‖_F` ∈ [0,1] | bounded, symmetric: the plotting axis |
| **interference ratio** | `max_k Σ_{j≠k} tr(A_kA_j) / tr(A_k²)` | interference over signal, per model |
| rearrangement floor | per-pair minimum by the rearrangement inequality | separates "solver stalled" from "collision unavoidable" |

**The restated prediction: the knee is at interference = 1.0** — 0 dB, where interference power
matches signal power in a model's own weighted frame. The max over `k`, not the mean, to match
worst-task accuracy.

The solver warm-starts by anti-aligning the spectra, which is *exactly optimal for N = 2*, and
keeps the best of several restarts. Without that it stalls well above the floor, and a solver
artifact would be indistinguishable from a capacity result.

### Two ways the prediction can be wrong

1. **Too permissive.** The metric is coherent-blind: `N` identical models score `N−1` yet merge
   perfectly. A real knee well below 1.0 is the signature.
2. **Wrong variable.** `C_j` is measured on model `j`'s data, but inside the merged model, model
   `j`'s machinery sees model `k`'s inputs. The honest term is a cross-covariance — `O(N²)` to
   measure, and it breaks the per-model structure. No knee anywhere implicates this.

---

## Phase 3 — Repair

Even a correct merge has wrong activation scales. Apply REPAIR-style renormalization: recompute
normalization statistics on calibration data and refit **only the RMSNorm gains** — a few thousand
parameters, no gradient through the body.

Always report pre- and post-repair numbers separately. Repair frequently recovers most of an
apparent failure, and conflating them hides which phase actually broke.

---

## Phase 4 — Evaluation

Grid: `N ∈ {2,4,8,16,32} × overlap ∈ {high, medium, disjoint} × d ∈ {128,256,512}`.

Reference points per cell:

- naive average — the floor
- each specialist on its own task — the ceiling
- a jointly trained model at the same budget
- a router over N specialists — the `N·P` bound
- **distillation of the N specialists into one student** — the honest strong competitor

Merging will not beat distillation on accuracy. It wins on being data-free and O(seconds). Say so
explicitly rather than omitting the comparison.

Report: per-task normalized accuracy, **worst-task accuracy** (means hide single-task collapse,
the dominant failure mode here), `Σr_k/d`, interpolation barrier, post-alignment subspace overlap.
Headline plot: degradation against `Σr_k/d`, with the predicted knee at 1.0.

---

## Phase 5 — Mechanistic verification

At this scale, on modular arithmetic, the circuits are directly readable — Fourier features for
modular addition are legible in the weights. When a merge degrades, check whether

- the addition circuit survived and the subtraction circuit was overwritten,
- the two superposed, or
- both were destroyed.

Nobody validates merging this way, because at production scale you cannot. It turns "it degraded
4%" into "here is the mechanism."

---

## Order of attack

Phase 0 plus Procrustes-on-embeddings, tested on same-task/different-seed pairs, is roughly two
weeks and tells you immediately whether the program is viable. **If the interpolation barrier does
not collapse there, stop and fix alignment before touching anything else.**
