#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MKL_ENABLE_INSTRUCTIONS="${MKL_ENABLE_INSTRUCTIONS:-SSE4_2}"

GAMMA_SCALE="${1:-20}"
REGULARIZER_SCALE="${2:-0.1}"
COMPLEXITY="${3:-complicated}"
RUNS="${4:-10}"

python3 main.py \
  --rounds 1000 \
  --runs "$RUNS" \
  --seed 0 \
  --high-probability \
  --complexity "$COMPLEXITY" \
  --practical-gamma-scale "$GAMMA_SCALE" \
  --practical-regularizer-scale "$REGULARIZER_SCALE"
