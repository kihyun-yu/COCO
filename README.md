# COCO Algorithm Implementations

This repository implements algorithms from the two papers in `./papers`:

- `1708.03741v1.pdf`: Yu, Neely, Wei (2017), Algorithm 1 for online convex optimization with stochastic constraints.
- `2606.31480v1.pdf`: Yu, Lee, Lee (2026), Algorithm 1 for constrained online convex optimization without Slater's condition.

The code is intentionally small and dependency-light. It requires NumPy and
uses tqdm to display experiment progress.

Install the dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Run tests

```bash
python3 -m unittest test_algorithms.py
```

## Run the gradual-Slater sweep

```bash
bash run/gradual_slater_sweep.sh
```

The sweep accepts practical COCO26 tuning parameters, a complexity mode, the
number of independent runs, the horizon, and the solution dimension:

```bash
bash run/gradual_slater_sweep.sh <gamma_scale> <regularizer_scale> <simple|complicated> <runs> <rounds> <dim>
```

For example:

```bash
bash run/gradual_slater_sweep.sh 200 0.01 simple 5 5000 5
```

For a single custom run, invoke the Python entry point directly:

```bash
python3 main.py --problem gradual-slater --rounds 2000 --seed 3
```

Progress is shown over all runs and rounds. Disable it when redirecting output
or running in automation:

```bash
python3 main.py --problem gradual-slater --no-progress
```

The comparison includes a practical tuned COCO26 curve. Its tuning knobs scale
the paper's Lyapunov parameter `gamma_t` and the dual regularizer:

```bash
python3 main.py --problem gradual-slater --practical-gamma-scale 20 --practical-regularizer-scale 0.1
```

The Python entry point defaults to a 5-dimensional solution space, a simple
quadratic objective with affine constraints, theorem-safe scaling, `gamma x1`, and
`regularizer x1`. The sweep defaults to the finite-horizon empirical setting
`gamma x200` and `regularizer x0.01`, but you can override both
through the positional arguments above. Dimensions from 2 through 10 are supported;
use `--complexity complicated` to opt into the composite objective.

The simple objective uses the standard unnormalized quadratic
`0.5 * ||x - center||^2`. The constraint random stream is kept independent of
dimension so changing the solution dimension does not reshuffle the realized
constraint sequence.

The gradual-Slater sweep defaults to the finite-horizon visualization tuning
`gamma x200` and `regularizer x0.01`. On the near-zero-margin 5D experiment,
this makes COCO2026 react more strongly to accumulated constraint violation and
keeps its violation curve below the two 2017 baselines. These are empirical
plotting defaults; the Python entry point retains the theorem-safe `x1` values.

This writes cumulative plots to `./result`:

```text
result/regret/regret_margin_*.jpg
result/violation/constraint_violation_margin_*.jpg
```

The plots compare three methods:

```text
Yu-Neely-Wei 2017 (known T): uses the final horizon T as an input parameter.
Yu-Neely-Wei 2017 (doubling): removes horizon knowledge by restarting at epochs 1, 2, 4, ...
Yu-Lee-Lee 2026 (practical): same primal-dual form with tuned finite-horizon constants.
```

Each curve is averaged over the configured independent runs. Shaded bands show
the pointwise 95% confidence interval.

Regret uses the exact empirical best fixed comparator for each run. The runner
first realizes all simple quadratic objectives, averages their centers, and
projects that realized mean onto the expected feasible set. Thus the comparator
uses the complete objective sequence and can differ across coordinates and
runs; it is not the population proxy `margin / dim`.

The comparison script uses a shared synthetic stochastic constraint stream designed to
highlight the advantage of algorithms that only require feasibility in expectation.
Use `--complexity simple` for scheduled quadratic losses and affine constraints,
or `--complexity complicated` for composite losses and nonlinear convex
constraints.
In the default simple setting, all coordinates of the quadratic base center
alternate together every 30 rounds between `0.8` and `0.2`. Each coordinate
then receives independent noise `Uniform(-0.10, 0.10)`. The resulting center
always lies in `[0.1, 0.9]^d`, so clipping is unnecessary. Constraint noise has the fixed distribution
`Uniform(-noise_magnitude, noise_magnitude)`; the gradual-Slater experiments use
`noise_magnitude=0.35`. Both noise sources are zero-mean, bounded, independent,
and have no time-varying scales or shock mixtures.

Each round samples one scalar nonlinear convex constraint. In the complicated
setting, the sampler is time-varying: probabilities drift smoothly and enter
short burst regimes, so the active constraint family is not drawn from one fixed
categorical distribution. Conceptually, the three realized constraint families
behave like:

```text
family 0: x[0] + 0.1*x[1]^2 <= 0.4
family 1: x[1] + 0.1*x[0]^2 <= 0.4
family 2: 0.85 - x[0] - x[1] + 0.1*(x[0]-x[1])^2 <= 0
```

If all three constraint types appear, the round-wise feasible region is empty:

```text
x[0] + 0.1*x[1]^2 <= 0.4 and x[1] + 0.1*x[0]^2 <= 0.4
imply x[0] + x[1] <= 0.8, which conflicts with the third constraint.
```

The long-run mixture is centered near the original stochastic-constraint
problem, whose expected constraint is feasible:

```text
E[g_t(x)] =
    0.35 * (x[0] + x[1])
  + 0.045 * (x[0]^2 + x[1]^2)
  + 0.01 * (x[0] - x[1])^2
  - 0.275 <= 0
```

The comparator is chosen on the boundary of this expected feasible set, so it
is feasible in expectation even though it is not round-wise feasible when the
rare third constraint appears. This is the setting targeted by stochastic-
constraint COCO algorithms.

## Gradual Slater Sweep

The gradual sweep uses

```text
E[g_t(x)] = sum_i x[i] + 0.08*sum_i x[i]^2 - margin <= 0
```

All decision coordinates participate in this constraint. The boundary
comparator distributes the margin symmetrically across all dimensions.

When `margin > 0`, Slater's condition holds with strict feasible point `x = 0`.
When `margin = 0`, Slater's condition fails. The sweep runs margins
`0.25, 0.10, 0.05, 0.02, 0.00` and writes regret/violation plots into
separate directories with margin-specific filenames:

```text
result/regret/regret_margin_*.jpg
result/violation/constraint_violation_margin_*.jpg
```
