#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

MODE="${1:-train}"              # train | resume | val | vis | check-data
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

DATA_ROOT="${DATA_ROOT:-/home/pengfei/HTCnet/DataSets}"
SAVE_DIR="${SAVE_DIR:-../Checkpoint_SUN_V1}"
PRETRAINED_ENCODER="${PRETRAINED_ENCODER:-/home/pengfei/HTCnet/Checkpoint/mit_b2.pth}"
RESUME_CKPT="${RESUME_CKPT:-$SAVE_DIR/latest_model.pth}"
VAL_CKPT="${VAL_CKPT:-$SAVE_DIR/best_model.pth}"

GPUS="${GPUS:-3}"
ENCODER_NAME="${ENCODER_NAME:-mit_b2}"
N_CLASSES="${N_CLASSES:-41}"
INPUT_HEIGHT="${INPUT_HEIGHT:-480}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-6e-5}"
EMA_DECAY="${EMA_DECAY:-0.9996}"
EMA_WARMUP_EPOCHS="${EMA_WARMUP_EPOCHS:-10}"
USE_DP="${USE_DP:-1}"
SAFE_MODE="${SAFE_MODE:-0}"
BATCH_IS_GLOBAL="${BATCH_IS_GLOBAL:-1}"
LABEL_DIR_NAME="${LABEL_DIR_NAME:-Labels}"
LABEL_MAP="${LABEL_MAP:-}"
IGNORE_INDEX="${IGNORE_INDEX:-0}"
MIOU_START_CLASS="${MIOU_START_CLASS:-1}"
VAL_INTERVAL="${VAL_INTERVAL:-2}"
DEBUG_VAL_STATS="${DEBUG_VAL_STATS:-0}"
NO_AMP="${NO_AMP:-0}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.08}"
LAMBDA_BOUNDARY="${LAMBDA_BOUNDARY:-0.03}"
LAMBDA_FEATURE_PRECISION="${LAMBDA_FEATURE_PRECISION:-0.03}"
HHA_EDGE_WEIGHT="${HHA_EDGE_WEIGHT:-0.0}"
MAX_EDGE_POS_WEIGHT="${MAX_EDGE_POS_WEIGHT:-10.0}"
OHEM_MIN_KEPT="${OHEM_MIN_KEPT:-30000}"
BOUNDARY_CE_WEIGHT="${BOUNDARY_CE_WEIGHT:-0.7}"
LOSS_WARMUP_START="${LOSS_WARMUP_START:-0.15}"
EDGE_WARMUP_EPOCHS="${EDGE_WARMUP_EPOCHS:-10}"
BOUNDARY_WARMUP_EPOCHS="${BOUNDARY_WARMUP_EPOCHS:-14}"
FEATURE_WARMUP_EPOCHS="${FEATURE_WARMUP_EPOCHS:-18}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SUN_ROOT="$DATA_ROOT/SUNRGBD"
REQUIRED_PATHS=(
  "$SUN_ROOT/RGB"
  "$SUN_ROOT/HHA"
  "$SUN_ROOT/$LABEL_DIR_NAME"
  "$SUN_ROOT/train.txt"
  "$SUN_ROOT/test.txt"
)
if [[ "$MODE" != "check-data" ]]; then
  for path in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "[run] ERROR: missing SUNRGBD path: $path"
      echo "[run] Expected structure: \$DATA_ROOT/SUNRGBD/{RGB,HHA,$LABEL_DIR_NAME,train.txt,test.txt}"
      exit 1
    fi
  done
fi

VAL_EXTRA_ARGS=()
if [[ "$DEBUG_VAL_STATS" == "1" ]]; then
  VAL_EXTRA_ARGS+=(--debug-val-stats)
fi
if [[ "$SAFE_MODE" == "1" ]]; then
  VAL_EXTRA_ARGS+=(--safe-mode)
fi

VIS_EXTRA_ARGS=()
if [[ "$SAFE_MODE" == "1" ]]; then
  VIS_EXTRA_ARGS+=(--safe-mode)
fi

LABEL_MAP_ARGS=()
if [[ -n "$LABEL_MAP" ]]; then
  LABEL_MAP_ARGS+=(--label-map "$LABEL_MAP")
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
REQUESTED_NUM_GPUS="${#GPU_ARRAY[@]}"
if command -v python >/dev/null 2>&1; then
  VISIBLE_NUM_GPUS="$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
)"
else
  VISIBLE_NUM_GPUS="$REQUESTED_NUM_GPUS"
fi
NUM_GPUS="$VISIBLE_NUM_GPUS"
if [[ "$USE_DP" == "1" && "$REQUESTED_NUM_GPUS" -gt 0 && "$VISIBLE_NUM_GPUS" -eq 0 ]]; then
  echo "[run] ERROR: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES but PyTorch sees 0 CUDA devices."
  echo "[run] Check whether the requested GPU ids are valid on this machine."
  exit 1
fi
if [[ "$REQUESTED_NUM_GPUS" -ne "$VISIBLE_NUM_GPUS" ]]; then
  echo "[run] WARNING: requested $REQUESTED_NUM_GPUS GPU(s) via GPUS=$GPUS, but PyTorch sees $VISIBLE_NUM_GPUS visible CUDA device(s)."
  echo "[run] Launching with $VISIBLE_NUM_GPUS process(es) to avoid invalid device ordinal."
fi
LOCAL_BATCH_SIZE="$BATCH_SIZE"
TOTAL_BATCH_SIZE="$BATCH_SIZE"
if [[ "$USE_DP" == "1" && "$NUM_GPUS" -gt 1 ]]; then
  if [[ "$BATCH_IS_GLOBAL" == "1" ]]; then
    if (( BATCH_SIZE < NUM_GPUS )); then
      echo "[run] ERROR: BATCH_SIZE=$BATCH_SIZE is smaller than NUM_GPUS=$NUM_GPUS when BATCH_IS_GLOBAL=1"
      exit 1
    fi
    if (( BATCH_SIZE % NUM_GPUS != 0 )); then
      echo "[run] ERROR: BATCH_SIZE=$BATCH_SIZE must be divisible by NUM_GPUS=$NUM_GPUS when BATCH_IS_GLOBAL=1"
      exit 1
    fi
    LOCAL_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))
  else
    TOTAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
  fi
fi

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --save-dir "$SAVE_DIR"
  --pretrained-encoder "$PRETRAINED_ENCODER"
  --n-classes "$N_CLASSES"
  --ignore-index "$IGNORE_INDEX"
  --miou-start-class "$MIOU_START_CLASS"
  --input-height "$INPUT_HEIGHT"
  --input-width "$INPUT_WIDTH"
  --batch-size "$LOCAL_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --epochs "$EPOCHS"
  --lr "$LR"
  --ema-decay "$EMA_DECAY"
  --ema-warmup-epochs "$EMA_WARMUP_EPOCHS"
  --label-dir-name "$LABEL_DIR_NAME"
  --val-interval "$VAL_INTERVAL"
  --encoder-name "$ENCODER_NAME"
  --lambda-edge "$LAMBDA_EDGE"
  --lambda-boundary "$LAMBDA_BOUNDARY"
  --lambda-feature-precision "$LAMBDA_FEATURE_PRECISION"
  --hha-edge-weight "$HHA_EDGE_WEIGHT"
  --max-edge-pos-weight "$MAX_EDGE_POS_WEIGHT"
  --ohem-min-kept "$OHEM_MIN_KEPT"
  --boundary-ce-weight "$BOUNDARY_CE_WEIGHT"
  --loss-warmup-start "$LOSS_WARMUP_START"
  --edge-warmup-epochs "$EDGE_WARMUP_EPOCHS"
  --boundary-warmup-epochs "$BOUNDARY_WARMUP_EPOCHS"
  --feature-warmup-epochs "$FEATURE_WARMUP_EPOCHS"
  --amp-dtype "$AMP_DTYPE"
)

if [[ -n "$LABEL_MAP" ]]; then
  COMMON_ARGS+=(--label-map "$LABEL_MAP")
fi

if [[ "$USE_DP" != "1" ]]; then
  COMMON_ARGS+=(--no-data-parallel)
fi

if [[ "$SAFE_MODE" == "1" ]]; then
  COMMON_ARGS+=(--safe-mode)
fi

if [[ "$DEBUG_VAL_STATS" == "1" ]]; then
  COMMON_ARGS+=(--debug-val-stats)
fi

if [[ "$NO_AMP" == "1" ]]; then
  COMMON_ARGS+=(--no-amp)
fi

TRAIN_LAUNCHER=(python train.py)
if [[ "$USE_DP" == "1" && "$NUM_GPUS" -gt 1 ]]; then
  TRAIN_LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" train.py)
fi

case "$MODE" in
  train)
    echo "[run] Training PFNet on GPUs: $CUDA_VISIBLE_DEVICES"
    echo "[run] Data root: $DATA_ROOT"
    echo "[run] Save dir: $SAVE_DIR"
    echo "[run] Batch per GPU: $LOCAL_BATCH_SIZE | Total batch: $TOTAL_BATCH_SIZE"
    "${TRAIN_LAUNCHER[@]}" "${COMMON_ARGS[@]}" 2>&1 | tee "logs/train_${TIMESTAMP}.log"
    ;;

  resume)
    echo "[run] Resuming PFNet from: $RESUME_CKPT"
    echo "[run] Batch per GPU: $LOCAL_BATCH_SIZE | Total batch: $TOTAL_BATCH_SIZE"
    "${TRAIN_LAUNCHER[@]}" "${COMMON_ARGS[@]}" --resume "$RESUME_CKPT" 2>&1 | tee "logs/resume_${TIMESTAMP}.log"
    ;;

  val)
    echo "[run] Validating PFNet checkpoint: $VAL_CKPT"
    python val.py \
      --data-root "$DATA_ROOT" \
      --ckpt "$VAL_CKPT" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --n-classes "$N_CLASSES" \
      --input-height "$INPUT_HEIGHT" \
      --input-width "$INPUT_WIDTH" \
      --label-dir-name "$LABEL_DIR_NAME" \
      --encoder-name "$ENCODER_NAME" \
      --use-tta \
      "${VAL_EXTRA_ARGS[@]}" \
      "${LABEL_MAP_ARGS[@]}" \
      2>&1 | tee "logs/val_${TIMESTAMP}.log"
    ;;

  vis)
    IMG_NAME="${2:-}"
    if [ -z "$IMG_NAME" ]; then
      echo "Usage: bash run.sh vis <image_full_name> [additional_args...]"
      echo "Example: bash run.sh vis 000014.jpg --plot-features"
      exit 1
    fi
    
    BASE_NAME="${IMG_NAME%.*}"
    shift 2 # 移除 mode 和 IMG_NAME，剩下的参数直接传给 python 脚本
    
    echo "[run] Visualizing with PFNet checkpoint: $VAL_CKPT"
    echo "[run] Target image: $IMG_NAME"
    
    python inference.py \
      --rgb "$DATA_ROOT/SUNRGBD/RGB/$IMG_NAME" \
      --hha "$DATA_ROOT/SUNRGBD/HHA/${BASE_NAME}.png" \
      --label "$DATA_ROOT/SUNRGBD/$LABEL_DIR_NAME/${BASE_NAME}.png" \
      --ckpt "$VAL_CKPT" \
      --n-classes "$N_CLASSES" \
      --input-height "$INPUT_HEIGHT" \
      --input-width "$INPUT_WIDTH" \
      --encoder-name "$ENCODER_NAME" \
      "${VIS_EXTRA_ARGS[@]}" \
      --save-path "logs/vis_${BASE_NAME}_${TIMESTAMP}.png" \
      "$@" \
      2>&1 | tee "logs/vis_${BASE_NAME}_${TIMESTAMP}.log"
    ;;

  check-data)
    shift
    echo "[run] Checking SUNRGBD dataset"
    echo "[run] Data root: $DATA_ROOT"
    python check_sunrgbd_dataset.py \
      --data-root "$DATA_ROOT" \
      --label-dir-name "$LABEL_DIR_NAME" \
      --n-classes "$N_CLASSES" \
      "$@" \
      2>&1 | tee "logs/check_data_${TIMESTAMP}.log"
    ;;

  *)
    echo "Usage: bash run.sh [train|resume|val|vis|check-data]"
    exit 1
    ;;
esac
