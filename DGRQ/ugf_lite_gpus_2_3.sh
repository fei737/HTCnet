#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Default launcher for running UGF-Lite experiments on physical GPUs 2 and 3.
# Usage:
#   bash ugf_lite_gpus_2_3.sh all-phase1
#   bash ugf_lite_gpus_2_3.sh auto-core
#   bash ugf_lite_gpus_2_3.sh all E5_full_lite

MODE="${1:-all-phase1}"
shift || true

export GPUS="${GPUS:-2,3}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-16}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export EPOCHS="${EPOCHS:-100}"

echo "[ugf-gpus-2-3] GPUS=$GPUS MODE=$MODE"
echo "[ugf-gpus-2-3] batch=$BATCH_SIZE target_global_batch=$TARGET_GLOBAL_BATCH epochs=$EPOCHS"

bash ugf_lite.sh "$MODE" "$@"
