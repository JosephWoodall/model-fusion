# Working in this repo

## The one rule

Every reparameterization must be **exactly function-preserving**. If a merge
loses accuracy, that loss has to be attributable to the merge and never to the
change of frame. `tests/test_symmetry.py` enforces this numerically, in float64,
and those tests are not optional. A change that loosens one of those tolerances
needs to say why in the commit message.

## Adding a symmetry

1. Implement it in `fusion/symmetry.py` as an in-place function returning the model.
2. Add an exactness test asserting the model's output is unchanged to `~1e-9` in float64.
3. Add it to `scramble()`, so every canonicalizer is tested against it.
4. Document the group and the argument for invariance in `docs/DESIGN.md`.

## Adding an alignment method

Put published baselines in `fusion/align/baselines.py` with a docstring naming
the paper and stating **which group it searches**. That last part is the whole
comparison: the point of this architecture is that it exposes `O(d)` where most
methods only have permutations.

## Reporting results

- Pre- and post-repair numbers are reported **separately**, always. Conflating
  them hides which phase broke, which defeats the design.
- The headline accuracy is **worst-task**, not mean. Means hide single-task
  collapse, which is the dominant failure mode in N-way merging.
- Distillation goes in the table even though merging loses to it. The claim for
  merging is data-free and O(seconds), not accuracy.
- A result that contradicts the capacity law goes in `docs/FINDINGS.md` with the
  command that produced it.

## Before pushing

```bash
pytest -q
ruff check src tests experiments
```
