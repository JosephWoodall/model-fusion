# Design invariants

The code is organized around one rule: **every reparameterization must be exactly
function-preserving**. If a merge loses accuracy, that loss must be attributable to the merge, never
to the change of frame. `tests/test_symmetry.py` enforces this numerically.

## Model

Decoder-only transformer, pre-norm, bias-free.

```
h ← E[tokens] + P[pos]
for each layer:
    h ← h + Attn(RMSNorm(h))
    h ← h + MLP(RMSNorm(h))
logits ← RMSNorm(h) @ U
```

Shapes (all weights are stored so that the residual dimension is the *leading* axis of a read and
the *trailing* axis of a write):

| Tensor | Shape | Role |
|--------|-------|------|
| `E` | `[V, d]` | writes to stream |
| `P` | `[T, d]` | writes to stream |
| `W_q, W_k, W_v` | `[d, n_h·d_h]` | read from stream |
| `W_o` | `[n_h·d_h, d]` | writes to stream |
| `W_in` | `[d, h]` | reads from stream |
| `W_out` | `[h, d]` | writes to stream |
| `U` | `[d, V]` | reads from stream |

RMSNorm carries **no learned gain** in the trained model. A gain vector exists only as an optional
field used by Phase 3 repair, and `fold_gains()` absorbs it into the following read-matrix so the
model returns to the gainless form the symmetry argument requires.

## The three symmetry groups

### 1. Residual stream — `O(d)`

For orthogonal `R`, `RMSNorm(hR) = RMSNorm(h)R` because `‖hR‖₂ = ‖h‖₂`. So

```
E ← E R      P ← P R
W_q,k,v ← Rᵀ W_q,k,v        W_o ← W_o R
W_in ← Rᵀ W_in              W_out ← W_out R
U ← Rᵀ U
```

leaves every logit unchanged. This is `fusion.symmetry.apply_residual_rotation`.

*This is the group the whole project rests on.* It is available only because the network is
gainless-RMSNorm and bias-free. A single bias anywhere in the residual path destroys it.

### 2. Attention heads — `GL(d_head)`

Attention scores depend on `W_q, W_k` only through the **QK circuit** `C_qk = W_q W_kᵀ ∈ R^{d×d}`,
and the head output depends on `W_v, W_o` only through the **OV circuit**
`C_ov = W_v W_o ∈ R^{d×d}`. Hence for any invertible `A`, `B` of size `d_head`:

```
W_q ← W_q A,   W_k ← W_k A^{−T}          (C_qk unchanged)
W_v ← W_v B,   W_o ← B^{−1} W_o          (C_ov unchanged)
```

The canonical representative is obtained from the rank-`d_head` SVD of each circuit:
`C = U S Vᵀ  ⟹  (W_q, W_k) := (U_r S_r^{1/2}, V_r S_r^{1/2})`, and likewise for OV. Two heads
implementing the same circuit land on the same representative regardless of the basis they were
trained in.

This group is *internal to a head*: it commutes with the residual rotation, so canonicalization
order is (1) residual, then (2) heads, then (3) head permutation.

### 3. MLP hidden units — permutation × positive diagonal

For a positively homogeneous activation (ReLU), `act(x s) = act(x) s` for `s > 0`, so

```
W_in ← W_in Π S,   W_out ← S^{−1} Πᵀ W_out
```

is function-preserving. GELU is **not** homogeneous, so under GELU only the permutation `Π` is
exact and scaling is approximate; the model defaults to ReLU for this reason and
`fusion.symmetry.apply_mlp_scaling` refuses to claim exactness under GELU.

## Canonicalization is per-model, never pairwise

Each model is mapped independently into a canonical frame. Consequences:

- N-way alignment costs N canonicalizations, not `N(N−1)/2` alignments.
- The result does not depend on the order the models are presented in.
- Adding an `N+1`-th model does not require redoing anything.

The one place a consensus target appears is the Procrustes iteration for the residual frame, where
`Ē` is the running consensus of the aligned embeddings. That iteration is initialized from the
mean and is invariant to model ordering up to the usual sign/rotation degeneracy of the consensus
itself.

## Separating the three failure modes

| If this breaks | You see |
|----------------|---------|
| Alignment | Control A (model vs. a rotated copy) fails to reach a zero barrier |
| Solution diversity | Control A passes, Control B (same task, different seeds) does not |
| Solver | Disentangled overlap sits above the rearrangement floor — nothing is attributable |
| Capacity | Solver at the floor, and merged accuracy falls as the interference ratio crosses 1 |
| Repair | Merged activations have wrong scale; post-repair accuracy jumps back |

The solver row matters as much as the others. An optimizer that stalls produces exactly the
signature of a capacity limit — high residual overlap — so `DisentangleResult.at_floor` gates
every capacity claim, and cells above the floor are drawn hollow in the headline plot.

Pre- and post-repair numbers are always reported separately, for exactly this reason.
