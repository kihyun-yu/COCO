# COCO Algorithm Implementations

This repository contains compact NumPy implementations and synthetic
experiments for:

- Yu, Neely, and Wei (2017), *Online Convex Optimization with Stochastic Constraints*
- Yu, Lee, and Lee (2026), *Constrained Online Convex Optimization without Slater's Condition*

The experiments compare three methods:

1. Yu–Neely–Wei 2017 with a known horizon.
2. An anytime doubling-epoch version of the 2017 method.
3. A practical finite-horizon tuning of Yu–Lee–Lee 2026.

## Setup

The project requires Python 3, NumPy, tqdm, and Matplotlib. Install the
dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Run the tests with:

```bash
python3 -m unittest test_algorithms.py
```

## Decision and loss model

The decision set is the unit box

```text
X = [0,1]^dim,
```

where `dim` can range from 2 through 10. Every algorithm starts at

```text
x_1 = (0.5, ..., 0.5).
```

At round `t`, the loss is

```text
f_t(x) = 0.5 * ||x - center_t||_2^2,
grad f_t(x) = x - center_t.
```

The center sequence is deterministic and supports two schedules.

### Complementary schedule

Select it with:

```bash
--loss-schedule complementary
```

In two dimensions, the center alternates between

```text
(0.8, 0.2) and (0.2, 0.8),
```

holding each value for `loss_switch_interval` rounds. In higher dimensions,
the schedule rotates the coordinate containing `0.8` and alternates every
center with its complement. A complete higher-dimensional cycle contains
`2 * dim` blocks.

### Sinusoidal schedule

Select it with:

```bash
--loss-schedule sinusoidal --loss-rotation-period 200
```

In two dimensions,

```text
theta_t = 2*pi*(t - 1) / loss_rotation_period
center_t = (
    0.5 + 0.3*cos(theta_t),
    0.5 + 0.3*sin(theta_t)
)
```

so the center moves continuously around a circle inside `[0.2,0.8]^2`. In
higher dimensions, coordinate `i` uses the phase
`theta_t + 2*pi*i/dim` in a cosine wave.

Both schedules keep every center coordinate between `0.2` and `0.8` and give
each coordinate a mean value of `0.5` over a complete cycle. The switch
interval is ignored by the sinusoidal schedule, and the rotation period is
ignored by the complementary schedule.

## Constraint problems

The CLI exposes three problems through `--problem`.

### `gradual-slater`

This is the primary sweep problem. Its stochastic constraint is

```text
g_t(x) = sum_i x[i] + noise_t - margin,
noise_t ~ Uniform(-0.35, 0.35).
```

Therefore,

```text
E[g_t(x)] = sum_i x[i] - margin,
```

and the expected feasible set is

```text
{x in [0,1]^dim : sum_i x[i] <= margin}.
```

- If `margin > 0`, Slater's condition holds because `x=0` is strictly feasible.
- If `margin = 0`, the expected feasible set is the singleton `{0}`, and
  Slater's condition fails.
- A small positive value such as `0.001` represents a nearly no-Slater case.

### `no-slater`

This is the exact zero-margin problem:

```text
g_t(x) = sum_i x[i] + noise_t.
```

Its expected feasible set is only `{0}`. A realized round has no feasible point
whenever its noise is positive.

### `slater`

This problem samples one affine constraint from three conflicting families:

```text
family 0: x[0] <= 0.4
family 1: x[1] <= 0.4
family 2: x[0] + x[1] >= 0.85
```

If all three families occur, their joint round-wise feasible region is empty.
The reference base mixture has probabilities `(0.45, 0.45, 0.10)` and expected
constraint

```text
E_base[g_t(x)] = 0.35 * (x[0] + x[1]) - 0.275.
```

The actual family probabilities drift over time and enter short burst regimes,
so this base-mixture expression is a reference constraint rather than every
round's conditional expectation.

## Algorithms and constants

The known-horizon 2017 method uses

```text
V = sqrt(T),
alpha = T.
```

The doubling version runs epochs of length `1, 2, 4, 8, ...`. It resets the
virtual queue and horizon-dependent parameters at each epoch while warm-starting
the next epoch from the previous decision.

The 2026 implementation computes its problem bounds before the run:

```text
L = max(sqrt(dim), sqrt(2)),
G = dim + 2,
D = diameter([0,1]^dim) = sqrt(dim).
```

These are fixed conservative bounds, not quantities learned from sampled data.
The practical scale arguments multiply the 2026 `gamma_t` and dual regularizer.
They are empirical experiment settings rather than changes to `L`, `G`, or `D`.

## Running `main.py`

Run one comparison directly with:

```bash
python3 main.py \
  --problem gradual-slater \
  --slater-margin 0.1 \
  --loss-schedule sinusoidal \
  --loss-rotation-period 200 \
  --rounds 2000 \
  --seed 3
```

Use `python3 main.py --help` for every available output and algorithm option.
The important defaults are:

| Option | Default | Meaning |
|---|---:|---|
| `--rounds` | `1000` | Rounds per run |
| `--runs` | `5` | Independent runs to average |
| `--dim` | `2` | Decision dimension |
| `--seed` | `0` | First run seed |
| `--problem` | `slater` | Constraint problem |
| `--slater-margin` | `0.0` | Margin for `gradual-slater` |
| `--loss-schedule` | `complementary` | Loss-center schedule |
| `--loss-switch-interval` | `30` | Complementary block length |
| `--loss-rotation-period` | `200` | Sinusoidal cycle length |
| `--practical-gamma-scale` | `1.0` | 2026 gamma multiplier |
| `--practical-regularizer-scale` | `1.0` | 2026 regularizer multiplier |

Progress reporting is enabled by default. Disable it for automation with:

```bash
python3 main.py --no-progress
```

Add `--high-probability` to use the delta-dependent 2026 potential and gamma
formula. The implementation uses `delta=0.05` unless configured directly in
Python.

## Gradual-Slater sweep

Run the four-margin sweep with:

```bash
bash run/gradual_slater_sweep.sh
```

Display its complete input guide with:

```bash
bash run/gradual_slater_sweep.sh --help
```

Its positional interface is:

```text
bash run/gradual_slater_sweep.sh \
  [gamma_scale] [regularizer_scale] [runs] [rounds] [dim] \
  [loss_switch_interval] [loss_schedule] [loss_rotation_period]
```

Current defaults:

| Position | Input | Default |
|---:|---|---:|
| 1 | `gamma_scale` | `300` |
| 2 | `regularizer_scale` | `0.05` |
| 3 | `runs` | `5` |
| 4 | `rounds` | `10000` |
| 5 | `dim` | `2` |
| 6 | `loss_switch_interval` | `20` |
| 7 | `loss_schedule` | `sinusoidal` |
| 8 | `loss_rotation_period` | `50` |

The script fixes the first seed at `0` and evaluates margins `0.7`, `0.5`,
`0.3`, and `0.1`. All of its outputs are kept under
`result/gradual_slater_sweep/`.

Example:

```bash
bash run/gradual_slater_sweep.sh 300 0.05 5 10000 2 20 sinusoidal 50
```

## Two-dimensional trajectory tracker

Run a focused 2D movement experiment with:

```bash
bash run/track_2d_trajectories.sh
```

Display its complete input guide with:

```bash
bash run/track_2d_trajectories.sh --help
```

Its positional interface is:

```text
bash run/track_2d_trajectories.sh \
  [gamma_scale] [regularizer_scale] [runs] [rounds] \
  [loss_switch_interval] [margin] [loss_schedule] [loss_rotation_period]
```

Current defaults:

| Position | Input | Default |
|---:|---|---:|
| 1 | `gamma_scale` | `200` |
| 2 | `regularizer_scale` | `0.01` |
| 3 | `runs` | `5` |
| 4 | `rounds` | `5000` |
| 5 | `loss_switch_interval` | `5` |
| 6 | `margin` | `0.5` |
| 7 | `loss_schedule` | `sinusoidal` |
| 8 | `loss_rotation_period` | `200` |

The tracker fixes `dim=2` and `seed=0`. To reproduce the previous
piecewise-constant schedule, run:

```bash
bash run/track_2d_trajectories.sh \
  200 0.01 5 5000 5 0.5 complementary 200
```

All of its outputs are kept under `result/track_2d_trajectories/`.

## Comparator, metrics, and plots

For every run, the experiment realizes the complete quadratic-loss sequence,
computes its weighted mean center, and projects that center onto the selected
problem's reference expected feasible set. This gives the exact empirical best
fixed comparator for the implemented loss sequence.

The plotted regret is

```text
sum_t f_t(x_t) - sum_t f_t(x_comparator).
```

Constraint violation is the cumulative signed sum

```text
sum_t g_t(x_t),
```

not the positive part of that sum.

For multiple runs, regret and violation curves show pointwise means with
normal-approximation 95% confidence intervals (`mean +/- 1.96 * standard
error`). Each algorithm receives a separate trajectory plot using the
pointwise mean decision and displaying one point every 20 rounds. A square
marks the start and an `X` marks the end.

For `dim=2`, the trajectory is the full path inside `[0,1]^2`. For higher
dimensions, the figure is the projection onto coordinates `x[0]` and `x[1]`.

The gradual sweep writes:

```text
result/gradual_slater_sweep/regret_margin_*.jpg
result/gradual_slater_sweep/constraint_violation_margin_*.jpg
result/gradual_slater_sweep/decision_trajectory_margin_*_yu2017_known_t.jpg
result/gradual_slater_sweep/decision_trajectory_margin_*_yu2017_doubling.jpg
result/gradual_slater_sweep/decision_trajectory_margin_*_yu2026.jpg
```

The focused tracker writes schedule-specific filenames beneath
`result/track_2d_trajectories/`, for example:

```text
result/track_2d_trajectories/decision_trajectory_sinusoidal_margin_0_5_yu2017_known_t.jpg
result/track_2d_trajectories/decision_trajectory_sinusoidal_margin_0_5_yu2017_doubling.jpg
result/track_2d_trajectories/decision_trajectory_sinusoidal_margin_0_5_yu2026.jpg
result/track_2d_trajectories/regret_sinusoidal_margin_0_5.jpg
result/track_2d_trajectories/constraint_violation_sinusoidal_margin_0_5.jpg
```

The gradual sweep filenames do not contain the schedule name, so running it
again with another schedule overwrites plots for the same margins.

## Reproducibility notes

- Loss centers are deterministic for a chosen schedule and horizon.
- Loss and constraint random generators are split from each run's seed.
- With multiple runs, seeds are `seed`, `seed+1`, and so on.
- Constraint sampling is separated from loss generation, so changing the
  decision dimension or loss schedule does not reshuffle the constraint stream
  for a fixed seed.
