#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MKL_ENABLE_INSTRUCTIONS="${MKL_ENABLE_INSTRUCTIONS:-SSE4_2}"

# Finite-horizon tuning that keeps COCO2026's cumulative violation below the
# 2017 baselines on the hardest near-zero-margin experiment.
GAMMA_SCALE="${1:-200}"
REGULARIZER_SCALE="${2:-0.01}"
COMPLEXITY="${3:-simple}"
RUNS="${4:-5}"
ROUNDS="${5:-5000}"
DIM="${6:-5}"

for margin in 0.25 0.10 0.05 0.02 0.0; do
  dir_margin="${margin/./_}"
  python3 main.py \
    --problem gradual-slater \
    --slater-margin "$margin" \
    --rounds "$ROUNDS" \
    --runs "$RUNS" \
    --dim "$DIM" \
    --seed 0 \
    --complexity "$COMPLEXITY" \
    --result-dir "result" \
    --regret-dir "result/regret" \
    --violation-dir "result/violation" \
    --regret-filename "regret_margin_${dir_margin}.jpg" \
    --violation-filename "constraint_violation_margin_${dir_margin}.jpg" \
    --practical-gamma-scale "$GAMMA_SCALE" \
    --practical-regularizer-scale "$REGULARIZER_SCALE"
done
