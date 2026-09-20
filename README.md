# model-fusion

Fusing *N* independently trained transformers into one model of the same size, from scratch,
with a falsifiable theory of when it can work.

This is a research testbed, not a library. Everything here is trained from scratch at a scale
where the resulting circuits are directly readable, so that a failed merge can be attributed to a
specific cause rather than reported as an unexplained accuracy drop.

## The framing

Most merging work conflates three separable problems, which is why its failures are ambiguous.
This repo builds a **separate diagnostic for each**, so a failure tells you which one broke.

| # | Problem | Statement | Diagnostic |
|---|---------|-----------|------------|
| 1 | **Alignment** | The models live in different frames. | Linear-interpolation barrier; layerwise CKA |
| 2 | **Capacity** | *N* feature sets must fit in one parameter budget. A packing problem, not an alignment problem. | Σ<sub>k</sub> r<sub>k</sub> / d; post-alignment subspace overlap |
| 3 | **Repair** | Merged activations have wrong statistics even when 1 and 2 succeed. | Pre- vs. post-repair accuracy, always reported separately |

The central claim under test:

> **A merge succeeds iff Σ<sub>k</sub> r<sub>k</sub> ≤ d**, where r<sub>k</sub> is the effective
> rank of model *k*'s residual-stream activations and *d* is the residual width.

That is a scaling law rather than a heuristic, and it generalizes over arbitrary *N* by
construction. The headline result is degradation plotted against Σr<sub>k</sub>/d, with a
predicted knee at 1.0.

## Why train from scratch

Choosing the architecture lets us *maximize the symmetry group available for alignment* — the
single biggest lever, and one you do not get when merging checkpoints someone else trained.

- **RMSNorm without a learned gain**, with the gain folded into the following linear layer. An
  RMSNorm network is computationally invariant under orthogonal transforms of the residual stream
  (the SliceGPT result). This upgrades the alignment group from permutations `P` to the full
  orthogonal group `O(d)` — a vastly richer set of alignments.
- **No biases.** Biases pin the frame and buy little at this scale.
- **Shared tokenizer and vocabulary across all N models.** Every embedding matrix is then
  expressed in a common token basis, which is the anchor the whole alignment hangs on.
- **Identical architecture, different seeds and data.** Weight merging has no meaning otherwise.

`fusion.symmetry` implements these transforms as exact, test-enforced invariances: applying any of
them leaves the model's function unchanged to floating-point tolerance. Any observed loss is
therefore attributable to the merge, never to the reparameterization.

## Canonicalization, not pairwise alignment

Pairwise alignment to a reference model is order-dependent and O(N²). Instead each model is mapped
*independently* into a canonical frame, which makes N-way alignment free and order-invariant.

| Component | Exploitable freedom | Method |
|-----------|--------------------|--------|
| Residual stream, on the anchor span | `O(d)` | Orthogonal Procrustes on `[E; P; Uᵀ]` — every matrix whose rows are in the shared token basis — iterating the consensus target |
| Residual stream, off it | `O(d − r)` | Diagonalize the residual activation covariance restricted to the complement; sign-fix by skewness |
| Attention heads | `GL(d_head)`, not just permutation | SVD of the QK circuit `W_Q W_Kᵀ` and the OV circuit `W_V W_O`; rotate to singular-vector coordinates, then Hungarian across heads |
| MLP hidden units | Permutation + positive scaling | Normalize rows to fix scale, match by activation correlation |

Head canonicalization is *exact*: attention depends on `W_Q` and `W_K` only through the QK circuit,
and on `W_V`, `W_O` only through the OV circuit, so rewriting each circuit in its own singular
basis changes no function value.

**The second row is not in the original plan and it is the difference between the method working
and not working.** Procrustes on the embeddings pins the frame only on the span of the anchor
matrices — at most `V + T` directions. At `d=128` with `V=51` that leaves 71 directions
undetermined, and *those directions carry 55% of the residual energy at layers 1 and 2*, because
attention and MLP outputs write wherever they like. The symptom is embedding disagreement falling
to `1e-17` while the interpolation barrier does not move. See
[`docs/FINDINGS.md`](docs/FINDINGS.md).

## Phases

| Phase | Content | Status |
|-------|---------|--------|
| 0 | Testbed: bias-free RMSNorm transformer, task suite with a relatedness dial, same-task/different-seed control | implemented |
| 1 | Canonicalization: anchor Procrustes + complement pinning, head and MLP canonicalization; baselines (Git Re-Basin, activation matching, OT/Sinkhorn, ZipIt!) | implemented, **validated** |
| 2 | Capacity-aware fusion: effective rank, Stiefel-manifold subspace disentangling, budgeted merge | implemented, knee **not yet located** |
| 3 | Repair: REPAIR-style renormalization, refitting only the RMSNorm gains | implemented |
| 4 | Evaluation grid and reference points | implemented |
| 5 | Mechanistic verification: read the Fourier features for modular arithmetic directly out of the merged weights | implemented |

See [`docs/PLAN.md`](docs/PLAN.md) for the full program and [`docs/DESIGN.md`](docs/DESIGN.md)
for the invariants the code is required to preserve.

## The go/no-go, in two controls

```bash
python -m experiments.exp0_sanity
```

One control is not enough, and finding that out is the first result this repo produced.

**Control A — pure frame difference.** A trained model against an exactly rotated copy of itself.
The two are the same function in different coordinates, so a working canonicalizer must drive the
interpolation barrier to *zero*. This tests the machinery and nothing else.

**Control B — same task, different seeds.** The real question.

Run on this testbed:

| | Control A | Control B |
|---|---|---|
| interpolation barrier | 4.74 → **0.000000** | 4.69 → 6.57 |
| worst parameter difference after alignment | 6e-6 (float32 noise) | — |

Control A passing while Control B does not is *not* an alignment failure, and the mechanistic
readout says what it is instead: the two seeds grokked **different Fourier frequencies**
(`[1,7,11,18,23]` vs `[1,13,16,22]`). They compute the same function through different circuits,
and no element of `O(d)` maps one onto the other. CKA agrees independently, and CKA cannot be
moved by alignment.

So the gate is:

- Control A fails → the aligner is broken. Fix it.
- Control A passes, Control B fails → solution diversity. That is a **capacity** question, and
  whether two circuits fit is exactly what `Σr_k ≤ d` decides.
- Both pass → proceed.

Run as originally specified — Control B alone — this test would have sent us to fix an aligner
that is provably exact.

## What the baselines are worth

`python -m experiments.exp1_alignment_baselines` — merged accuracy after each alignment method,
then averaging. Chance is 0.021.

| Method | Known frame difference | Independent seeds |
|--------|-----------------------|-------------------|
| naive average (floor) | 0.053 | 0.025 |
| Git Re-Basin (weight matching) | 0.022 | 0.020 |
| activation matching | 0.050 | 0.023 |
| OT fusion (Sinkhorn) | 0.035 | 0.023 |
| ZipIt! feature merging | 0.044 | 0.023 |
| **canonicalization (this repo)** | **1.000** | 0.024 |

Left column: every method is handed a difference it can in principle undo. Only the one that
searches `O(d)` undoes it. Every permutation-based method stays at chance, because the difference
is not a permutation. That gap is the concrete value of choosing the architecture for its symmetry
group.

Right column: nothing recovers anything, for the Control B reason above. Reported rather than
omitted.

## Measuring capacity without a magic number

The capacity law needs a well-defined `r_k`. A thresholded effective rank is not one — measured
here on a single modular-addition model at `d=128`:

| energy threshold | 0.90 | 0.99 | 0.999 |
|---|---|---|---|
| effective rank | 39 | 98 | 124 |

The threshold picks the answer, and with it the predicted knee. The **participation ratio**
`(Σλ)² / Σλ²` has no such knob and lands at ~11 for the same model, which is where the energy
actually is. It is the default (`FusionConfig.rank_measure`); the thresholded version is kept for
comparison against the literature.

## Evaluation is adversarial on purpose

Every cell of the grid `N ∈ {2,4,8,16,32} × overlap ∈ {high,medium,disjoint} × d ∈ {128,256,512}`
is reported against five reference points:

- **naive weight average** — the floor
- **each specialist on its own task** — the ceiling
- **a jointly trained model at the same budget**
- **a router over N specialists** — the N·P bound
- **distillation of the N specialists into one student** — the honest strong competitor

Merging will not beat distillation on accuracy. It wins on being data-free and O(seconds). That
comparison is stated explicitly rather than omitted.

Reported metrics: per-task normalized accuracy, **worst-task accuracy** (means hide single-task
collapse, which is the dominant failure mode here), Σr<sub>k</sub>/d, interpolation barrier, and
post-alignment subspace overlap.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q            # invariance tests: the symmetry transforms must be exact
```

## Results so far

[`docs/FINDINGS.md`](docs/FINDINGS.md) is the running log, with the command for every number.
Short version: Phase 1 is validated and exact; Phase 2's knee has not been located yet, because
every cell run so far is already over capacity and the sweep needs to start from a feasible one.
Nothing here yet shows a merge that works on independently trained models.

## Layout

```
src/fusion/
  model.py          bias-free RMSNorm transformer
  symmetry.py       the exact reparameterizations (O(d), GL(d_head), MLP perm+scale)
  tasks.py          task suite with a relatedness dial
  train.py          specialist training
  align/            canonicalization + published baselines
  capacity/         effective rank, Stiefel disentangling, budgeted merge
  repair.py         REPAIR-style gain refitting
  evaluation/       barrier, CKA, overlap, the grid harness
  baselines/        distillation, router, joint training
  mechanistic.py    Fourier-circuit readout: superposed / overwritten / destroyed
experiments/        runnable phase-by-phase scripts
docs/               PLAN.md (the program), DESIGN.md (invariants), FINDINGS.md (results)
```

## References

Ainsworth et al., *Git Re-Basin* · Ashmore & Gashler / Singh & Jaggi, *OT fusion* ·
Stoica et al., *ZipIt!* · Jordan et al., *REPAIR* · Ashkboos et al., *SliceGPT* ·
Nanda et al., *Progress measures for grokking via mechanistic interpretability*

## License

MIT
