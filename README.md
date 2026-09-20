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
| Residual stream | `O(d)` | Orthogonal Procrustes on the shared-vocabulary embeddings, iterating the consensus target |
| Attention heads | `GL(d_head)`, not just permutation | SVD of the QK circuit `W_Q W_Kᵀ` and the OV circuit `W_V W_O`; rotate to singular-vector coordinates, then Hungarian across heads |
| MLP hidden units | Permutation + positive scaling | Normalize rows to fix scale, match by activation correlation |

Head canonicalization is *exact*: attention depends on `W_Q` and `W_K` only through the QK circuit,
and on `W_V`, `W_O` only through the OV circuit, so rewriting each circuit in its own singular
basis changes no function value.

## Phases

| Phase | Content | Status |
|-------|---------|--------|
| 0 | Testbed: bias-free RMSNorm transformer, task suite with a relatedness dial, same-task/different-seed control | implemented |
| 1 | Canonicalization: Procrustes, head and MLP canonicalization; baselines (Git Re-Basin, activation matching, OT/Sinkhorn, ZipIt!) | implemented |
| 2 | Capacity-aware fusion: effective rank, Stiefel-manifold subspace disentangling, budgeted merge | implemented |
| 3 | Repair: REPAIR-style renormalization, refitting only the RMSNorm gains | implemented |
| 4 | Evaluation grid and reference points | implemented |
| 5 | Mechanistic verification: read the Fourier features for modular arithmetic directly out of the merged weights | implemented |

See [`docs/PLAN.md`](docs/PLAN.md) for the full program and [`docs/DESIGN.md`](docs/DESIGN.md)
for the invariants the code is required to preserve.

## Order of attack

Phase 0 plus Procrustes-on-embeddings, tested on **same-task/different-seed** pairs, is the
go/no-go. If the interpolation barrier does not collapse there, alignment is broken and nothing
downstream means anything — stop and fix it before touching capacity or repair.

```bash
python -m experiments.exp0_sanity          # train 2 same-task models, align, check the barrier
```

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
experiments/        runnable phase-by-phase scripts
```

## References

Ainsworth et al., *Git Re-Basin* · Ashmore & Gashler / Singh & Jaggi, *OT fusion* ·
Stoica et al., *ZipIt!* · Jordan et al., *REPAIR* · Ashkboos et al., *SliceGPT* ·
Nanda et al., *Progress measures for grokking via mechanistic interpretability*

## License

MIT
