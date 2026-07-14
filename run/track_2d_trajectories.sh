#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MKL_ENABLE_INSTRUCTIONS="${MKL_ENABLE_INSTRUCTIONS:-SSE4_2}"

usage() {
  cat <<'EOF'
Usage:
  bash run/track_2d_trajectories.sh \
    [gamma_scale] [regularizer_scale] [runs] [rounds] \
    [loss_switch_interval] [margin] [loss_schedule] [loss_rotation_period]

Inputs:
  gamma_scale             Multiplier for the 2026 algorithm's gamma_t.
                          Default: 200
  regularizer_scale       Multiplier for the 2026 dual regularizer.
                          Default: 0.01
  runs                    Number of independent runs to average.
                          Default: 5
  rounds                  Number of online rounds in each run.
                          Default: 5000
  loss_switch_interval    Rounds per center in the complementary schedule.
                          Ignored by the sinusoidal schedule. Default: 5
  margin                  Slater margin in the expected constraint.
                          Default: 0.5
  loss_schedule           Loss-center schedule: complementary or sinusoidal.
                          Default: sinusoidal
  loss_rotation_period    Rounds per full sinusoidal rotation.
                          Ignored by the complementary schedule. Default: 200

The decision dimension is fixed at 2, the seed is fixed at 0, and each
algorithm receives a separate trajectory plot sampled every 20 rounds.
Results are written beneath result/track_2d_trajectories/.

Example:
  bash run/track_2d_trajectories.sh 200 0.01 5 5000 5 0.5 sinusoidal 200

Previous complementary setting:
  bash run/track_2d_trajectories.sh 200 0.01 5 5000 5 0.5 complementary 200
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

GAMMA_SCALE="${1:-200}"
REGULARIZER_SCALE="${2:-0.01}"
RUNS="${3:-5}"
ROUNDS="${4:-5000}"
LOSS_SWITCH_INTERVAL="${5:-5}"
MARGIN="${6:-0.5}"
LOSS_SCHEDULE="${7:-sinusoidal}"
LOSS_ROTATION_PERIOD="${8:-200}"
DIR_MARGIN="${MARGIN/./_}"
OUTPUT_DIR="result/track_2d_trajectories"

python3 main.py \
  --problem gradual-slater \
  --slater-margin "$MARGIN" \
  --rounds "$ROUNDS" \
  --runs "$RUNS" \
  --dim 2 \
  --loss-switch-interval "$LOSS_SWITCH_INTERVAL" \
  --loss-schedule "$LOSS_SCHEDULE" \
  --loss-rotation-period "$LOSS_ROTATION_PERIOD" \
  --seed 0 \
  --result-dir "$OUTPUT_DIR" \
  --regret-dir "$OUTPUT_DIR" \
  --violation-dir "$OUTPUT_DIR" \
  --trajectory-dir "$OUTPUT_DIR" \
  --regret-filename "regret_${LOSS_SCHEDULE}_margin_${DIR_MARGIN}.jpg" \
  --violation-filename "constraint_violation_${LOSS_SCHEDULE}_margin_${DIR_MARGIN}.jpg" \
  --trajectory-filename "decision_trajectory_${LOSS_SCHEDULE}_margin_${DIR_MARGIN}.jpg" \
  --practical-gamma-scale "$GAMMA_SCALE" \
  --practical-regularizer-scale "$REGULARIZER_SCALE"
