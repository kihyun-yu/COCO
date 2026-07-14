#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MKL_ENABLE_INSTRUCTIONS="${MKL_ENABLE_INSTRUCTIONS:-SSE4_2}"

usage() {
  cat <<'EOF'
Usage:
  bash run/gradual_slater_sweep.sh \
    [gamma_scale] [regularizer_scale] [runs] [rounds] [dim] \
    [loss_switch_interval] [loss_schedule] [loss_rotation_period]

Inputs:
  gamma_scale             Multiplier for the 2026 algorithm's gamma_t.
                          Default: 300
  regularizer_scale       Multiplier for the 2026 dual regularizer.
                          Default: 0.05
  runs                    Number of independent runs to average.
                          Default: 5
  rounds                  Number of online rounds in each run.
                          Default: 10000
  dim                     Decision dimension; must be between 2 and 10.
                          Default: 2
  loss_switch_interval    Rounds per center in the complementary schedule.
                          Ignored by the sinusoidal schedule. Default: 20
  loss_schedule           Loss-center schedule: complementary or sinusoidal.
                          Default: sinusoidal
  loss_rotation_period    Rounds per full sinusoidal rotation.
                          Ignored by the complementary schedule. Default: 50

The script runs margins 0.7, 0.5, 0.3, and 0.1 with seed 0.
Each algorithm receives a separate trajectory plot sampled every 20 rounds.
Results are written beneath result/gradual_slater_sweep/.

Example:
  bash run/gradual_slater_sweep.sh 300 0.05 5 10000 2 20 sinusoidal 50
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Finite-horizon tuning that keeps COCO2026's cumulative violation below the
# 2017 baselines on the hardest near-zero-margin experiment.
GAMMA_SCALE="${1:-300}"
REGULARIZER_SCALE="${2:-0.001}"
RUNS="${3:-10}"
ROUNDS="${4:-10000}"
DIM="${5:-2}"
LOSS_SWITCH_INTERVAL="${6:-20}"
LOSS_SCHEDULE="${7:-sinusoidal}"
LOSS_ROTATION_PERIOD="${8:-50}"
OUTPUT_DIR="result/gradual_slater_sweep"

for margin in 0.7 0.5 0.3 0.1; do
  dir_margin="${margin/./_}"
  python3 main.py \
    --problem gradual-slater \
    --slater-margin "$margin" \
    --rounds "$ROUNDS" \
    --runs "$RUNS" \
    --dim "$DIM" \
    --loss-switch-interval "$LOSS_SWITCH_INTERVAL" \
    --loss-schedule "$LOSS_SCHEDULE" \
    --loss-rotation-period "$LOSS_ROTATION_PERIOD" \
    --seed 0 \
    --result-dir "$OUTPUT_DIR" \
    --regret-dir "$OUTPUT_DIR" \
    --violation-dir "$OUTPUT_DIR" \
    --trajectory-dir "$OUTPUT_DIR" \
    --regret-filename "regret_margin_${dir_margin}.jpg" \
    --violation-filename "constraint_violation_margin_${dir_margin}.jpg" \
    --trajectory-filename "decision_trajectory_margin_${dir_margin}.jpg" \
    --practical-gamma-scale "$GAMMA_SCALE" \
    --practical-regularizer-scale "$REGULARIZER_SCALE"
done
