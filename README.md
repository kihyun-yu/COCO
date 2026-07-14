# COCO: Gradual-Slater Experiments

This repository compares three constrained online convex optimization methods
as Slater's condition becomes weaker:

- Yu, Neely, and Wei (2017), with a known horizon
- an anytime doubling-epoch version of Yu et al. (2017)
- Yu, Lee, and Lee (2026), without Slater's condition

## Setup

Python 3 is required. Install the dependencies and run the tests with:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest test_algorithms.py
```

## Gradual-Slater problem

Decisions lie in the unit box `X = [0,1]^dim`. At round `t`, the stochastic
constraint is

```text
g_t(x) = sum_i x[i] + noise_t - margin,
noise_t ~ Uniform(-0.35, 0.35).
```

Its expected feasible set is

```text
{x in [0,1]^dim : sum_i x[i] <= margin}.
```

As `margin` approaches zero, strict feasibility disappears. At `margin = 0`,
the expected feasible set is the singleton `{0}` and Slater's condition fails.

The quadratic loss is

```text
f_t(x) = 0.5 * ||x - center_t||_2^2.
```

The default sinusoidal schedule moves `center_t` continuously through the
decision space.

## Run the sweep

Run the standard experiment with:

```bash
bash run/gradual_slater_sweep.sh
```

It evaluates margins `0.7`, `0.5`, `0.3`, and `0.1`, using seed `0`. Results
are written to `result/gradual_slater_sweep/`.

The optional positional arguments are:

```text
bash run/gradual_slater_sweep.sh \
  [gamma_scale] [regularizer_scale] [runs] [rounds] [dim] \
  [loss_switch_interval] [loss_schedule] [loss_rotation_period]
```

| Argument | Default |
|---|---:|
| `gamma_scale` | `300` |
| `regularizer_scale` | `0.001` |
| `runs` | `10` |
| `rounds` | `10000` |
| `dim` | `2` |
| `loss_switch_interval` | `20` |
| `loss_schedule` | `sinusoidal` |
| `loss_rotation_period` | `50` |

For example:

```bash
bash run/gradual_slater_sweep.sh 300 0.001 10 10000 2 20 sinusoidal 50
```

Use `bash run/gradual_slater_sweep.sh --help` for the full argument guide.

## Run one setting

To test a single margin directly:

```bash
python3 main.py \
  --problem gradual-slater \
  --slater-margin 0.1 \
  --loss-schedule sinusoidal \
  --loss-rotation-period 50 \
  --practical-gamma-scale 300 \
  --practical-regularizer-scale 0.001 \
  --rounds 10000 \
  --runs 10 \
  --seed 0
```

Use `python3 main.py --help` to see all output and algorithm options.

## Outputs

For each margin, the sweep saves:

```text
regret_margin_*.jpg
constraint_violation_margin_*.jpg
decision_trajectory_margin_*_<algorithm>.jpg
```

Regret is measured against the best fixed feasible decision for the realized
loss sequence. Constraint violation is the cumulative signed sum
`sum_t g_t(x_t)`. With multiple runs, plots show the pointwise mean and a 95%
normal-approximation confidence interval.

The loss sequence is deterministic for a chosen schedule and horizon. Run
seeds are consecutive (`seed`, `seed + 1`, ...), and constraint sampling is
kept separate from loss generation for reproducibility.
