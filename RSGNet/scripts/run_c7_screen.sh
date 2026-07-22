#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${RUN_ID:-pfhr_c7screen_seed3407}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-3407}"
C5_CKPT="${C5_CKPT:-$PROJECT_ROOT/../runs_rsgnet/C5_factorized_routed/pfhr_seed3407/best_miou_0.4866_epoch_100.pth}"

if [[ ! -f "$C5_CKPT" ]]; then
  echo "[C7 screen] C5 checkpoint not found: $C5_CKPT" >&2
  exit 1
fi

run_one() {
  local gpus="$1"
  local experiment="$2"
  local log_file="$PROJECT_ROOT/../runs_rsgnet/${experiment}/${RUN_ID}/screen_${gpus//,/}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$(dirname "$log_file")"

  (
    export GPUS="$gpus"
    export RUN_ID EPOCHS BATCH_SIZE TARGET_GLOBAL_BATCH NUM_WORKERS SEED
    export RESUME_CKPT="$C5_CKPT"
    export HHA_DEGRADE_PROB="0.35"
    export HHA_DEGRADE_MODES="dropout,noise,shift"
    export RELIABILITY_CLEAN_WEIGHT="0.20"
    export RELIABILITY_TARGET_TEMPERATURE="0.15"
    export LAMBDA_RELIABILITY="0.05"
    export USE_TTA_EVAL="0"

    echo "[C7 screen] train | experiment=$experiment | gpus=$GPUS | epochs=$EPOCHS"
    bash scripts/experiments.sh finetune "$experiment"

    echo "[C7 screen] eval | experiment=$experiment | gpus=$GPUS"
    bash scripts/experiments.sh eval "$experiment"

    echo "[C7 screen] routes | experiment=$experiment | gpus=$GPUS"
    bash scripts/experiments.sh routes "$experiment"
  ) 2>&1 | tee "$log_file"
}

run_one "0,1" "C7_final_only_reliability" &
PID_C7=$!
run_one "2,3" "C8_final_only_soft_reliability" &
PID_C8=$!

status=0
wait "$PID_C7" || status=1
wait "$PID_C8" || status=1

if [[ "$status" -ne 0 ]]; then
  echo "[C7 screen] one or more jobs failed; inspect runs_rsgnet/C7_final_only_reliability and C8_final_only_soft_reliability." >&2
  exit "$status"
fi

echo "[C7 screen] complete"
echo "[C7 screen] C7 metrics: $PROJECT_ROOT/../runs_rsgnet/C7_final_only_reliability/$RUN_ID/metrics.json"
echo "[C7 screen] C8 metrics: $PROJECT_ROOT/../runs_rsgnet/C8_final_only_soft_reliability/$RUN_ID/metrics.json"
