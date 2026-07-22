#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH=. \
GPUS="${GPUS:-0,1,2,3}" \
RUN_ID="${RUN_ID:-val_b5_rare_crop_seed3407}" \
ENCODER_NAME=mit_b5 \
PRETRAINED_ENCODER=/home/pengfei/HTCnet/Checkpoint/mit_b5.pth \
EPOCHS="${EPOCHS:-60}" \
SEED="${SEED:-3407}" \
BATCH_SIZE="${BATCH_SIZE:-2}" \
TARGET_GLOBAL_BATCH=16 \
NUM_WORKERS="${NUM_WORKERS:-4}" \
LR="${LR:-6e-5}" \
ENCODER_LR_MULT="${ENCODER_LR_MULT:-0.10}" \
DROP_PATH_RATE="${DROP_PATH_RATE:-0.20}" \
VAL_FRACTION="${VAL_FRACTION:-0.10}" \
VAL_SEED="${VAL_SEED:-2105}" \
VAL_INTERVAL=5 \
EARLY_STOPPING_PATIENCE=0 \
RARE_CROP_PROBABILITY="${RARE_CROP_PROBABILITY:-0.70}" \
RARE_CROP_CLASS_IDS="14,19,27,31" \
RARE_CROP_TRIALS=8 \
CLASS_WEIGHT_MODE=inverse_log \
CLASS_WEIGHT_CLAMP=4.0 \
AUG_LEVEL=base \
  bash scripts/experiments.sh train C5_factorized_routed
