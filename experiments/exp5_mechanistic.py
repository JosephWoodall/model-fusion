"""Phase 5: say *what* the merge did to the circuits, not just how much it cost.

Trains a modular-addition and a modular-subtraction specialist, merges them, and
reads the Fourier structure out of the merged weights. The verdict is one of
three concrete mechanisms rather than a percentage:

* **superposed** — both specialists' key frequencies survive at reduced amplitude
* **overwritten** — one specialist's frequencies survive, the other's are gone
* **destroyed** — neither survives

Nobody validates merging this way because at production scale you cannot read
the circuits. Here you can.

    python -m experiments.exp5_mechanistic [--steps 2000]
"""

from __future__ import annotations

import argparse

from fusion.align import canonicalize_all
from fusion.capacity import fuse, naive_average
from fusion.config import FusionConfig, ModelConfig, TrainConfig
from fusion.evaluation.metrics import per_task_accuracy
from fusion.mechanistic import diagnose_merge, fourier_profile, neuron_frequency_map
from fusion.repair import repair_statistics
from fusion.tasks import make_tasks, max_seq_len
from fusion.train import resolve_device, train_family


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=int, default=47)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--tasks", nargs="+", default=["mod_add", "mod_sub"])
    args = ap.parse_args()

    device = resolve_device()
    tasks = make_tasks(args.tasks, p=args.p)
    cfg = ModelConfig(vocab_size=tasks[0].vocab.size, d_model=args.d, n_layers=2, n_heads=4,
                      max_seq_len=max_seq_len(tasks))
    fcfg = FusionConfig()
    print(f"device: {device}\nconfig: {cfg}\n")

    specs = train_family(tasks, model_cfg=cfg, train_cfg=TrainConfig(steps=args.steps, log=False))
    models = [s.model for s in specs]
    by_name = {s.task.name: s.model for s in specs}
    calib = [t.batch(fcfg.calib_batch_size, device=device)[0] for t in tasks]

    print("--- specialists ---")
    for s in specs:
        print(f"  {s.name}: acc {s.final_accuracy():.4f}")
        print(f"    {fourier_profile(s.model, args.p)}")
        freqs = neuron_frequency_map(s.model, 0, args.p)
        print(f"    layer-0 neurons span {len(set(freqs.tolist()))} distinct Fourier rows")

    canon, _ = canonicalize_all(models, fcfg, calib_tokens=calib)
    merged, cap = fuse(canon, calib, fcfg, canonicalized=True)
    repaired = merged.clone()
    repair_statistics(repaired, models, calib[0])

    print(f"\n--- merge ---\n  {cap}")
    for label, model in (("naive average", naive_average(models)),
                         ("capacity fusion", merged),
                         ("capacity fusion + repair", repaired)):
        accs = per_task_accuracy(model, tasks)
        diag = diagnose_merge(model, by_name, args.p)
        print(f"\n  {label}")
        print(f"    accuracy : {  {k: round(v, 3) for k, v in accs.items()} }")
        print(f"    circuits : {diag}")
        print(f"    merged key frequencies: {diag.merged_keys}")
        for name, keys in diag.specialist_keys.items():
            print(f"      {name} needed {keys}")

    print("\nThis is the payoff of the from-scratch testbed: an accuracy drop comes with a")
    print("mechanism. 'Overwritten' and 'destroyed' call for different fixes -- the first")
    print("is a packing failure, the second is an alignment or repair failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
