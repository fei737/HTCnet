#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

EPOCHS="${EPOCHS:-60}"
SEED="${SEED:-3407}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
VAL_SEED="${VAL_SEED:-2105}"

run_one() {
  local gpus="$1"
  local run_id="$2"
  local rare_probability="$3"
  local log_dir="$PROJECT_ROOT/../runs_rsgnet/C5_factorized_routed/$run_id"
  mkdir -p "$log_dir"
  (
    PYTHONPATH=. \
    GPUS="$gpus" \
    RUN_ID="$run_id" \
    EPOCHS="$EPOCHS" \
    SEED="$SEED" \
    BATCH_SIZE=4 \
    TARGET_GLOBAL_BATCH=16 \
    NUM_WORKERS=4 \
    VAL_FRACTION="$VAL_FRACTION" \
    VAL_SEED="$VAL_SEED" \
    VAL_INTERVAL=5 \
    EARLY_STOPPING_PATIENCE=0 \
    RARE_CROP_PROBABILITY="$rare_probability" \
    RARE_CROP_CLASS_IDS="14,19,27,31" \
    RARE_CROP_TRIALS=8 \
    CLASS_WEIGHT_MODE=inverse_log \
    CLASS_WEIGHT_CLAMP=4.0 \
    AUG_LEVEL=base \
      bash scripts/experiments.sh train C5_factorized_routed
  ) 2>&1 | tee "$log_dir/screen.log"
}

run_one "0,1" "val_b2_control_seed${SEED}" "0.0" &
PID_CONTROL=$!
run_one "2,3" "val_b2_rare_crop_seed${SEED}" "0.70" &
PID_RARE=$!

status=0
wait "$PID_CONTROL" || status=1
wait "$PID_RARE" || status=1
exit "$status"
