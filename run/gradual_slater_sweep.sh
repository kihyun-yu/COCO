#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MKL_ENABLE_INSTRUCTIONS="${MKL_ENABLE_INSTRUCTIONS:-SSE4_2}"

# Finite-horizon tuning that keeps COCO2026's cumulative violation below the
# 2017 baselines on the hardest near-zero-margin experiment.
GAMMA_SCALE="${1:-320}"
REGULARIZER_SCALE="${2:-0.001}"
COMPLEXITY="${3:-simple}"
RUNS="${4:-10}"
ROUNDS="${5:-10000}"
DIM="${6:-5}"

for margin in 0.25 0.10 0.05 0.02 0.0001; do
  dir_margin="${margin/./_}"
  python3 main.py \
    --problem gradual-slater \
    --slater-margin "$margin" \
    --rounds "$ROUNDS" \
    --runs "$RUNS" \
    --dim "$DIM" \
    --seed 0 \
    --complexity "$COMPLEXITY" \
    --result-dir "result/gradual_slater" \
    --regret-dir "result/gradual_slater/regret" \
    --violation-dir "result/gradual_slater/violation" \
    --regret-filename "regret_margin_${dir_margin}.svg" \
    --violation-filename "constraint_violation_margin_${dir_margin}.svg" \
    --normalized-regret-filename "normalized_regret_margin_${dir_margin}.svg" \
    --normalized-violation-filename "normalized_constraint_violation_margin_${dir_margin}.svg" \
    --practical-gamma-scale "$GAMMA_SCALE" \
    --practical-regularizer-scale "$REGULARIZER_SCALE"
done
