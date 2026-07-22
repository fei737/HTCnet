#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${RUN_ID:-pfhr_c5ft60_seed3407}"
GPUS="${GPUS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="$PROJECT_ROOT/../runs_rsgnet/C5_factorized_routed/$RUN_ID"

for item in \
  "clean HHA_PFHR" \
  "dropout_p30 HHA_PFHR_dropout_p30" \
  "noise_m050 HHA_PFHR_noise_m050" \
  "shift_px08 HHA_PFHR_shift_px08" \
  "tilt_deg05 HHA_PFHR_tilt_deg05"
do
  read -r tag hha_dir <<< "$item"
  echo "[C5-ft60 physical] tag=$tag hha_dir=$hha_dir"
  GPUS="$GPUS" \
  RUN_ID="$RUN_ID" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  USE_TTA_EVAL=1 \
  FACTORIZED_HHA_DIR_NAME="$hha_dir" \
  EVAL_HHA_DEGRADE_MODE=clean \
  EVAL_HHA_DEGRADE_SEVERITY=0 \
  METRICS_JSON="$OUTPUT_DIR/physical_${tag}.json" \
    bash scripts/experiments.sh eval C5_factorized_routed
done

echo "[C5-ft60 physical] complete: $OUTPUT_DIR"
