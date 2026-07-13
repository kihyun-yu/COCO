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

## Run Tests

```bash
bash run/test.sh
```

## Compare Algorithms

```bash
bash run/default_experiment.sh
```

The experiment scripts accept practical COCO26 tuning parameters, a complexity
mode, and the number of independent runs to average:

```bash
bash run/default_experiment.sh <gamma_scale> <regularizer_scale> <simple|complicated> <runs> <dim>
bash run/high_probability_experiment.sh <gamma_scale> <regularizer_scale> <simple|complicated> <runs> <dim>
bash run/gradual_slater_sweep.sh <gamma_scale> <regularizer_scale> <simple|complicated> <runs> <rounds> <dim>
```

For example:

```bash
bash run/default_experiment.sh 10 0.25 simple 10 8
bash run/default_experiment.sh 20 0.1 complicated 10 10
bash run/gradual_slater_sweep.sh 320 0.001 simple 10 10000 5
```

You can also pass custom arguments through the main runner:

```bash
bash run/main.sh --rounds 2000 --seed 3
```

Progress is shown over all runs and rounds. Disable it when redirecting output
or running in automation:

```bash
bash run/main.sh --no-progress
```

The comparison includes a practical tuned COCO26 curve. Its tuning knobs scale
the paper's Lyapunov parameter `gamma_t` and the dual regularizer:

```bash
bash run/main.sh --practical-gamma-scale 20 --practical-regularizer-scale 0.1
```

The Python entry point defaults to a 5-dimensional solution space, a simple
quadratic objective with affine constraints, theorem-safe scaling, `gamma x1`, and
`regularizer x1`. The provided experiment scripts default to the finite-horizon
empirical setting `gamma x20` and `regularizer x0.1`, but you can override both
through the positional arguments above. Dimensions from 2 through 10 are supported;
use `--complexity complicated` to opt into the composite objective.

For dimensions above 2, the simple quadratic is normalized to the original 2D
scale, the comparator uses the long-run center on extra coordinates, and the
constraint random stream is kept independent of dimension. These choices make
the default 5D plots comparable to the original experiment.

For the high-probability parameter choice of the 2026 algorithm:

```bash
bash run/high_probability_experiment.sh
```

To see Slater's condition vanish gradually:

```bash
bash run/gradual_slater_sweep.sh
```

The gradual-Slater sweep defaults to the finite-horizon visualization tuning
`gamma x320` and `regularizer x0.001`. On the near-zero-margin 5D experiment,
this makes COCO2026 react more strongly to accumulated constraint violation and
keeps its violation curve below the two 2017 baselines. These are empirical
plotting defaults; the Python entry point retains the theorem-safe `x1` values.

This writes cumulative and normalized plots to `./result`:

```text
result/regret.svg
result/constraint_violation.svg
result/normalized_regret.svg
result/normalized_constraint_violation.svg
```

The plots compare three methods:

```text
Yu-Neely-Wei 2017 (known T): uses the final horizon T as an input parameter.
Yu-Neely-Wei 2017 (doubling): removes horizon knowledge by restarting at epochs 1, 2, 4, ...
Yu-Lee-Lee 2026 (practical): same primal-dual form with tuned finite-horizon constants.
```

Each curve is the average over 10 independent runs by default. Shaded bands show
the pointwise 95% confidence interval. The normalized plots divide cumulative
regret and cumulative signed violation by the round index, which is useful for
long-horizon convergence checks.

The comparison script uses a shared synthetic stochastic constraint stream designed to
highlight the advantage of algorithms that only require feasibility in expectation.
Use `--complexity simple` for scheduled quadratic losses and affine constraints,
or `--complexity complicated` for composite losses and nonlinear convex
constraints.
The loss functions are composite convex objectives: a nonstationary quadratic,
a log-sum-exp of affine pieces, and a smooth absolute-value penalty. The
quadratic center switches regimes frequently, uses faster oscillations, has
random jitter, and sometimes jumps to a spike regime. Constraint intercepts
include zero-mean bounded mixture noise: most rounds use moderate shocks, while
a minority of rounds use a wider shock radius. This makes the regret and
violation confidence bands less artificially narrow while keeping every
objective convex and every constraint noise term centered at zero.

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
result/gradual_slater/regret/regret_margin_*.svg
result/gradual_slater/regret/normalized_regret_margin_*.svg
result/gradual_slater/violation/constraint_violation_margin_*.svg
result/gradual_slater/violation/normalized_constraint_violation_margin_*.svg
```
