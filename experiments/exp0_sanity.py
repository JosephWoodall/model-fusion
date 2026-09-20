"""Phase 0 + Phase 1 go/no-go, in two controls.

**Control A — pure frame difference.** A trained model against an exactly
rotated copy of itself. The two are the same function in different coordinates,
so a working canonicalizer must recover the rotation and drive the
interpolation barrier to *zero*. This tests the machinery and nothing else. If
it fails, the alignment code is broken.

**Control B — same task, different seeds.** Two models trained on identical
data from different initializations. They differ by a frame *and* by whatever
genuinely different solution each found. This is the real question: if
canonicalization cannot collapse this barrier, nothing downstream — capacity,
repair, the scaling law — means anything.

Running both is what separates "the aligner is broken" from "the models learned
different things", which are otherwise indistinguishable from a single number.

**If Control A does not collapse, fix the aligner. If Control A collapses and
Control B does not, the residual gap is a solution-diversity problem, not an
alignment one — and that is a Phase 2 question.**

    python -m experiments.exp0_sanity [--steps 3000] [--d 128]
"""

from __future__ import annotations

import argparse

import torch

from fusion.align import anchor_basis, canonicalize_all, complement_conditioning
from fusion.align.procrustes import frame_disagreement
from fusion.capacity.merge import average_models, naive_average
from fusion.config import FusionConfig, ModelConfig, TrainConfig
from fusion.evaluation.metrics import interpolation_barrier, layerwise_cka
from fusion.mechanistic import fourier_profile
from fusion.symmetry import apply_residual_rotation, random_orthogonal
from fusion.tasks import accuracy, make_tasks
from fusion.train import resolve_device, train_specialist


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mod_add")
    ap.add_argument("--p", type=int, default=47)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--barrier-threshold", type=float, default=0.1)
    args = ap.parse_args()

    device = resolve_device()
    print(f"device: {device}")

    task = make_tasks([args.task], p=args.p)[0]
    cfg = ModelConfig(
        vocab_size=task.vocab.size, d_model=args.d, n_layers=args.layers,
        n_heads=args.heads, max_seq_len=task.seq_len,
    )
    print(f"config: {cfg}\n")

    print("training two models on the SAME task with DIFFERENT seeds")
    specs = [
        train_specialist(cfg, task, TrainConfig(steps=args.steps, seed=s))
        for s in args.seeds
    ]
    models = [s.model for s in specs]
    for s in specs:
        print(f"  {s.name}: acc {s.final_accuracy():.4f}")

    tokens, _ = task.batch(512, torch.Generator().manual_seed(0), device=device)

    # ---- Control A: the machinery test -------------------------------------
    print("\n=== CONTROL A: pure frame difference (must go to zero) ===")
    rotated = models[0].clone()
    apply_residual_rotation(
        rotated, random_orthogonal(args.d, seed=999).to(device=device, dtype=models[0].embed.dtype)
    )
    a_before = interpolation_barrier(models[0], rotated, task)["barrier"]
    canon_a, report_a = canonicalize_all([models[0], rotated], FusionConfig(),
                                         calib_tokens=tokens)
    a_after = interpolation_barrier(canon_a[0], canon_a[1], task)["barrier"]
    worst_param = max(
        (p.data - q.data).abs().max().item()
        for p, q in zip(canon_a[0].parameters(), canon_a[1].parameters(), strict=True)
    )
    print(f"  {report_a}")
    print(f"  barrier {a_before:.4f} -> {a_after:.6f}")
    print(f"  worst parameter difference after canonicalization: {worst_param:.3e}")
    control_a_ok = a_after < 1e-3
    print(f"  {'PASS' if control_a_ok else 'FAIL -- the aligner is broken; stop here'}")

    # ---- Control B: the real question --------------------------------------
    print("\n=== CONTROL B: same task, different seeds ===")
    print("\n--- before alignment ---")
    b_before = interpolation_barrier(models[0], models[1], task)
    cka_before = layerwise_cka(models[0], models[1], tokens)
    avg_before = accuracy(naive_average(models), task, device=device)
    print(f"  interpolation barrier : {b_before['barrier']:.4f} "
          f"(worst at alpha={b_before['argmax_alpha']:.1f})")
    print(f"  layerwise CKA         : {[f'{c:.3f}' for c in cka_before]}")
    print(f"  embedding disagreement: {frame_disagreement(models):.4e}")
    print(f"  naive-average accuracy: {avg_before:.4f}")

    canon, report = canonicalize_all(models, FusionConfig(), calib_tokens=tokens)

    print("\n--- after canonicalization ---")
    b_after = interpolation_barrier(canon[0], canon[1], task)
    cka_after = layerwise_cka(canon[0], canon[1], tokens)
    avg_after = accuracy(average_models(canon), task, device=device)
    print(f"  {report}")
    print(f"  interpolation barrier : {b_after['barrier']:.4f} "
          f"(worst at alpha={b_after['argmax_alpha']:.1f})")
    print(f"  layerwise CKA         : {[f'{c:.3f}' for c in cka_after]}")
    print(f"  aligned-average acc   : {avg_after:.4f}")

    # the specialists themselves must be untouched by alignment
    for m, s in zip(canon, specs, strict=True):
        assert abs(accuracy(m, task, device=device) - s.final_accuracy()) < 0.02, \
            "canonicalization changed a model's accuracy -- it is not function-preserving"

    basis = anchor_basis(canon)
    cond = complement_conditioning(canon[0], tokens, basis.to(canon[0].embed))
    print(f"  complement            : dim {cond['complement_dim']} of {args.d}, "
          f"{100 * cond['energy_fraction']:.1f}% of residual energy, "
          f"min eigen-gap {cond['min_relative_gap']:.3f}")

    print("\n--- what did each model actually learn? ---")
    for spec in specs:
        print(f"  {spec.name}: {fourier_profile(spec.model, args.p)}")

    print("\n--- verdict ---")
    reduction = 1 - b_after["barrier"] / max(b_before["barrier"], 1e-9)
    print(f"  barrier {b_before['barrier']:.4f} -> {b_after['barrier']:.4f} "
          f"({100 * reduction:.1f}% reduction)")
    print(f"  merged accuracy {avg_before:.4f} -> {avg_after:.4f} "
          f"(specialists: {specs[0].final_accuracy():.4f}, {specs[1].final_accuracy():.4f})")

    ok = b_after["barrier"] < args.barrier_threshold
    if not control_a_ok:
        print("\n  NO-GO. Control A failed: the canonicalizer cannot even undo a rotation")
        print("  it was handed. This is an implementation defect, not a research finding.")
        return 1
    if ok:
        print("\n  GO. Alignment collapses the barrier on both controls. Proceed to Phase 2.")
        return 0

    print(f"\n  PARTIAL. Control A passes ({a_after:.2e}), so the aligner works, but")
    print(f"  Control B's barrier is still {b_after['barrier']:.4f} "
          f"(threshold {args.barrier_threshold}).")
    print("  The two models are NOT related by a change of frame. Check, in order:")
    print("    1. the Fourier profiles above -- if the two models grokked *different*")
    print("       frequencies, no element of O(d) maps one onto the other, and the gap")
    print("       is solution diversity, not misalignment.")
    print("    2. layerwise CKA after alignment: well below 1 says the same thing")
    print("       independently, and CKA is rotation-invariant so alignment cannot move it.")
    print("    3. only if both look aligned should you suspect the canonicalizer.")
    print("  Solution diversity is a Phase 2 (capacity) question, not a Phase 1 one:")
    print("  two different circuits need two sets of directions, and whether they fit")
    print("  is exactly what sum(r_k) <= d decides.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
