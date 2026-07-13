# COCO Algorithm Implementations

This repository contains compact NumPy implementations of Algorithm 1 from:

- Yu, Neely, and Wei (2017), *Online Convex Optimization with Stochastic Constraints* (`papers/1708.03741v1.pdf`)
- Yu, Lee, and Lee (2026), *Constrained Online Convex Optimization without Slater's Condition* (`papers/2606.31480v1.pdf`)

The experiments compare the 2017 method with a known horizon, an anytime
doubling-trick variant of that method, and a practical finite-horizon tuning of
the 2026 method.

## Setup

The project requires Python 3, NumPy, tqdm, and Matplotlib. Install the Python
dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Tests

Run the unit tests with:

```bash
python3 -m unittest test_algorithms.py
```

## Running an experiment

Run one comparison directly through the Python entry point:

```bash
python3 main.py --problem gradual-slater --rounds 2000 --seed 3
```

By default, `main.py` uses:

- 1,000 rounds and 5 independent runs
- a 5-dimensional decision space (dimensions 2 through 10 are supported)
- the `slater` conflicting-constraint problem
- simple quadratic losses and affine constraints
- `gamma x1` and `regularizer x1` for the 2026 method
- progress reporting across all runs and rounds

Disable progress reporting in automation or when redirecting output:

```bash
python3 main.py --problem gradual-slater --no-progress
```

The practical 2026 tuning parameters multiply the paper's `gamma_t` value and
adaptive dual regularizer, respectively:

```bash
python3 main.py \
  --problem gradual-slater \
  --practical-gamma-scale 20 \
  --practical-regularizer-scale 0.1
```

The command-line parser exposes `--complexity complicated`, and the repository
contains generators for composite losses and nonlinear constraints. However,
the comparison runner currently computes regret using an exact comparator that
only supports simple quadratic losses. Consequently, use
`--complexity simple` when running `main.py` or the sweep; complicated mode
currently exits with a comparator error.

## Gradual-Slater sweep

Run the complete sweep with:

```bash
bash run/gradual_slater_sweep.sh
```

Its positional arguments are:

```text
bash run/gradual_slater_sweep.sh \
  <gamma_scale> <regularizer_scale> simple <runs> <rounds> <dim>
```

For example:

```bash
bash run/gradual_slater_sweep.sh 200 0.01 simple 5 5000 5
```

The sweep defaults to `gamma x200`, `regularizer x0.01`, 5 runs, 5,000
rounds, and 5 dimensions. It evaluates margins
`0.25, 0.10, 0.05, 0.02, 0.00`.

In the supported simple mode, the stochastic constraint is

```text
g_t(x) = sum_i x[i] + noise_t - margin,
noise_t ~ Uniform(-0.35, 0.35).
```

Therefore,

```text
E[g_t(x)] = sum_i x[i] - margin.
```

Every decision coordinate participates in the constraint. When `margin > 0`,
Slater's condition holds because `x = 0` is strictly feasible in expectation.
When `margin = 0`, the expected feasible set is the singleton `{0}`, so
Slater's condition fails.

The `gamma x200`, `regularizer x0.01` values are empirical visualization
settings, not the paper's unscaled constants. In the default 5D, 5-run,
5,000-round, seed-0 experiment at margin 0, the resulting 2026 cumulative
violation curve stays below both 2017 baselines. Other settings are not
guaranteed to preserve that ordering.

## Outputs and plotted metrics

The sweep writes JPEG plots to:

```text
result/regret/regret_margin_*.jpg
result/violation/constraint_violation_margin_*.jpg
```

The three plotted methods are:

```text
Yu-Neely-Wei 2017 (known T)
Yu-Neely-Wei 2017 (doubling epochs 1, 2, 4, ...)
Yu-Lee-Lee 2026 (practical finite-horizon tuning)
```

Each curve is the pointwise average over the configured independent runs. When
more than one run is used, the shaded band is the normal-approximation 95%
confidence interval (`mean +/- 1.96 * standard error`). Constraint violation is
plotted as the cumulative signed sum `sum_t g_t(x_t)`.

For every simple-mode run, regret is measured against the exact empirical best
fixed comparator. The runner realizes the complete quadratic-loss sequence,
computes the mean of its centers, and projects that mean onto the reference
expected feasible set. The resulting comparator can differ across coordinates;
it is generally not the symmetric population proxy `margin / dim`.

## Simple loss sequence

The supported simple loss is

```text
f_t(x) = 0.5 * ||x - center_t||^2.
```

All coordinates of the center move together in a deterministic 60-round cycle:
30 rounds at `0.8`, followed by 30 rounds at `0.2`. There is no randomness in
`center_t`.

Constraint sampling consumes the same number of random values at every
supported dimension, so changing the solution dimension does not reshuffle the
realized constraint sequence for a fixed seed.

## Conflicting stochastic-constraint problem

The `slater` problem samples one scalar constraint from three conflicting
families. In simple mode, the families are affine versions of:

```text
family 0: x[0] <= 0.4
family 1: x[1] <= 0.4
family 2: x[0] + x[1] >= 0.85
```

If all three types occur, their joint round-wise feasible region is empty. A
fixed reference mixture with probabilities `(0.45, 0.45, 0.10)` instead gives

```text
E_base[g_t(x)] = 0.35 * (x[0] + x[1]) - 0.275.
```

The actual sampler is deliberately time-varying: its probabilities drift and
enter short burst regimes. Thus the expression above is a reference constraint
for the base mixture, not the exact conditional expectation at every round or
the exact finite-horizon average distribution. The empirical comparator is
projected onto this reference feasible set.

For completeness, the currently non-runnable complicated comparison generator
uses nonlinear families of the form:

```text
family 0: x[0] + 0.1*x[1]^2 <= 0.4
family 1: x[1] + 0.1*x[0]^2 <= 0.4
family 2: 0.85 - x[0] - x[1] + 0.1*(x[0]-x[1])^2 <= 0
```

Under the fixed base mixture, their reference expected constraint is

```text
E_base[g_t(x)] =
    0.35 * (x[0] + x[1])
  + 0.045 * (x[0]^2 + x[1]^2)
  + 0.01 * (x[0] - x[1])^2
  - 0.275.
```

Because the implemented sampler is time-varying, a point on the boundary of
this base-mixture constraint is not necessarily feasible for every round's
conditional expectation or for the sampler's exact finite-horizon average.
