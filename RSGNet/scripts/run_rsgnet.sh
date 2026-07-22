#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

default_var() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    printf -v "$name" '%s' "$value"
  fi
}

MODE="${1:-train}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
default_var DATA_ROOT "/home/pengfei/HTCnet/DataSets"
default_var DATASET_NAME "SUNRGBD"
default_var PROTOCOL "sunrgbd37"
default_var SAVE_DIR "$PROJECT_ROOT/../runs_rsgnet/manual"
default_var PRETRAINED_ENCODER "/home/pengfei/HTCnet/Checkpoint/mit_b2.pth"
default_var RESUME_CKPT "$SAVE_DIR/latest_model.pth"
default_var METRICS_JSON "$SAVE_DIR/metrics.json"
default_var ROUTE_JSON "$SAVE_DIR/route_diagnostics.json"
default_var ROUTE_MAX_SAMPLES "500"
default_var GPUS "2,3"
default_var ENCODER_NAME "mit_b2"
default_var N_CLASSES "41"
default_var INPUT_HEIGHT "480"
default_var INPUT_WIDTH "480"
default_var ALLOW_CUSTOM_INPUT_SIZE "0"
default_var BATCH_SIZE "8"
default_var TARGET_GLOBAL_BATCH "16"
default_var NUM_WORKERS "8"
default_var EPOCHS "100"
default_var SEED "3407"
default_var LR "8e-5"
default_var ENCODER_LR_MULT "0.3"
default_var MIN_LR_RATIO "0.05"
default_var AMP_DTYPE "bf16"

default_var LABEL_DIR_NAME ""
default_var HHA_DIR_NAME "HHA"
default_var HHA_NAME_PREFIX ""
default_var HHA_CHANNEL_ORDER "dha"
default_var GEOMETRY_SOURCE "hha"
default_var DEPTH_DIR_NAME "Depth"
default_var DEPTH_NAME_PREFIX ""
default_var DEPTH_SCALE "1000.0"
default_var MAX_DEPTH "10.0"
default_var LABEL_NAME_PREFIX ""
default_var LABEL_MAP ""
default_var SPLIT_POLICY "file"
default_var IGNORE_INDEX "0"
default_var MIOU_START_CLASS "1"
default_var TTA_SCALES ""
default_var EVAL_NATIVE_SIZE "0"
default_var SLIDING_EVAL "0"
default_var EVAL_HHA_DEGRADE_MODE "clean"
default_var EVAL_HHA_DEGRADE_SEVERITY "0.0"
default_var EVAL_HHA_DEGRADE_SEED "3407"
default_var USE_TTA_EVAL "1"

if [[ -z "$LABEL_DIR_NAME" ]]; then
  LABEL_DIR_NAME="Labels"
  [[ "$DATASET_NAME" == "NYU" ]] && LABEL_DIR_NAME="Label"
fi
if [[ "$DATASET_NAME" == "NYU" && -z "$HHA_NAME_PREFIX" ]]; then
  HHA_NAME_PREFIX="hha_"
fi
if [[ "$PROTOCOL" == "sunrgbd37" ]]; then
  N_CLASSES=37
  if [[ "$ALLOW_CUSTOM_INPUT_SIZE" != "1" ]]; then
    INPUT_HEIGHT=480
    INPUT_WIDTH=480
  fi
  IGNORE_INDEX=255
  MIOU_START_CLASS=0
  SPLIT_POLICY="sunrgbd_official"
  EVAL_NATIVE_SIZE=1
  SLIDING_EVAL=1
  [[ -z "$TTA_SCALES" ]] && TTA_SCALES="0.5,0.75,1.0,1.25,1.5"
  [[ -z "$LABEL_MAP" ]] && LABEL_MAP="$PROJECT_ROOT/configs/sunrgbd40_to_37.json"
else
  [[ -z "$TTA_SCALES" ]] && TTA_SCALES="0.75,1.0,1.25"
fi

default_var FUSION_MODE "stagewise"
default_var ARCHITECTURE_VARIANT "legacy"
default_var GEOMETRY_ENCODING "factorized_routed"
default_var GEOMETRY_CHANNELS "dha"
default_var DISABLE_RELIABILITY "0"
default_var PROMPT_CHANNELS "32"
default_var DECODER_CHANNELS "256"
default_var ATTENTION_TOKENS "512"
default_var LOCAL_SCALE_INIT "0.05"
default_var SEMANTIC_SCALE_INIT "0.05"
default_var DROP_PATH_RATE "0.10"
default_var TRAIN_CUTOUT_PROB "0.10"
default_var AUG_LEVEL "base"
default_var HHA_DEGRADE_PROB "0.35"
default_var HHA_DEGRADE_MODES "dropout,noise,shift"
default_var RELIABILITY_CLEAN_WEIGHT "0.05"
default_var RELIABILITY_TARGET_TEMPERATURE "0.08"
default_var LAMBDA_RELIABILITY "0.05"
default_var RELIABILITY_WARMUP_EPOCHS "10"
default_var VAL_FRACTION "0.0"
default_var VAL_SEED "3407"
default_var RARE_CROP_PROBABILITY "0.0"
default_var RARE_CROP_CLASS_IDS ""
default_var RARE_CROP_TRIALS "8"
default_var USE_CLASS_WEIGHTS "1"
default_var CLASS_WEIGHT_MODE "inverse_log"
default_var CLASS_WEIGHT_CLAMP "4.0"
default_var CE_ONLY_EPOCHS "5"
default_var OHEM_START_EPOCH "100000"
default_var OHEM_MIN_KEPT "80000"
default_var LAMBDA_LOVASZ "0.30"
default_var LAMBDA_DICE "0.0"
default_var LAMBDA_EDGE "0.05"
default_var LAMBDA_BOUNDARY "0.0"
default_var LAMBDA_FEATURE_PRECISION "0.0"
default_var FEATURE_PRECISION_MAX_SIZE "160"
default_var AUX_WEIGHT "0.10"
default_var EARLY_STOPPING_PATIENCE "15"
default_var EARLY_STOPPING_MIN_EPOCH "30"
default_var VAL_INTERVAL "1"

if [[ "$DISABLE_RELIABILITY" == "1" ]]; then
  LAMBDA_RELIABILITY=0.0
  HHA_DEGRADE_PROB=0.0
fi

LOG_DIR="$SAVE_DIR/logs"
mkdir -p "$SAVE_DIR" "$LOG_DIR"
DATASET_ROOT="$DATA_ROOT/$DATASET_NAME"
GEOMETRY_DIR_NAME="$HHA_DIR_NAME"
[[ "$GEOMETRY_SOURCE" == "depth" ]] && GEOMETRY_DIR_NAME="$DEPTH_DIR_NAME"
for path in "$DATASET_ROOT/RGB" "$DATASET_ROOT/$GEOMETRY_DIR_NAME" \
  "$DATASET_ROOT/$LABEL_DIR_NAME" "$DATASET_ROOT/train.txt" "$DATASET_ROOT/test.txt"; do
  if [[ ! -e "$path" ]]; then
    echo "[RSGNet] Missing dataset path: $path" >&2
    exit 1
  fi
done

default_var PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"
default_var TORCH_NCCL_ASYNC_ERROR_HANDLING "1"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF
export TORCH_NCCL_ASYNC_ERROR_HANDLING
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_GPUS="${#GPU_ARRAY[@]}"

ARCH_ARGS=(
  --encoder-name "$ENCODER_NAME"
  --prompt-channels "$PROMPT_CHANNELS"
  --decoder-channels "$DECODER_CHANNELS"
  --attention-tokens "$ATTENTION_TOKENS"
  --local-scale-init "$LOCAL_SCALE_INIT"
  --semantic-scale-init "$SEMANTIC_SCALE_INIT"
  --fusion-mode "$FUSION_MODE"
  --architecture-variant "$ARCHITECTURE_VARIANT"
  --geometry-encoding "$GEOMETRY_ENCODING"
  --geometry-channels "$GEOMETRY_CHANNELS"
)
[[ "$DISABLE_RELIABILITY" == "1" ]] && ARCH_ARGS+=(--disable-reliability)

DATA_ARGS=(
  --data-root "$DATA_ROOT"
  --dataset-name "$DATASET_NAME"
  --protocol "$PROTOCOL"
  --n-classes "$N_CLASSES"
  --label-map "$LABEL_MAP"
  --split-policy "$SPLIT_POLICY"
  --ignore-index "$IGNORE_INDEX"
  --miou-start-class "$MIOU_START_CLASS"
  --input-height "$INPUT_HEIGHT"
  --input-width "$INPUT_WIDTH"
  --label-dir-name "$LABEL_DIR_NAME"
  --hha-dir-name "$HHA_DIR_NAME"
  --hha-name-prefix "$HHA_NAME_PREFIX"
  --hha-channel-order "$HHA_CHANNEL_ORDER"
  --geometry-source "$GEOMETRY_SOURCE"
  --depth-dir-name "$DEPTH_DIR_NAME"
  --depth-name-prefix "$DEPTH_NAME_PREFIX"
  --depth-scale "$DEPTH_SCALE"
  --max-depth "$MAX_DEPTH"
  --label-name-prefix "$LABEL_NAME_PREFIX"
)

TRAIN_ARGS=(
  "${DATA_ARGS[@]}"
  "${ARCH_ARGS[@]}"
  --save-dir "$SAVE_DIR"
  --pretrained-encoder "$PRETRAINED_ENCODER"
  --batch-size "$BATCH_SIZE"
  --target-global-batch "$TARGET_GLOBAL_BATCH"
  --num-workers "$NUM_WORKERS"
  --epochs "$EPOCHS"
  --seed "$SEED"
  --lr "$LR"
  --encoder-lr-mult "$ENCODER_LR_MULT"
  --warmup-epochs 5
  --min-lr-ratio "$MIN_LR_RATIO"
  --amp-dtype "$AMP_DTYPE"
  --train-cutout-prob "$TRAIN_CUTOUT_PROB"
  --aug-level "$AUG_LEVEL"
  --hha-degrade-prob "$HHA_DEGRADE_PROB"
  --hha-degrade-modes "$HHA_DEGRADE_MODES"
  --reliability-clean-weight "$RELIABILITY_CLEAN_WEIGHT"
  --reliability-target-temperature "$RELIABILITY_TARGET_TEMPERATURE"
  --lambda-reliability "$LAMBDA_RELIABILITY"
  --reliability-warmup-epochs "$RELIABILITY_WARMUP_EPOCHS"
  --val-fraction "$VAL_FRACTION"
  --val-seed "$VAL_SEED"
  --rare-crop-probability "$RARE_CROP_PROBABILITY"
  --rare-crop-class-ids "$RARE_CROP_CLASS_IDS"
  --rare-crop-trials "$RARE_CROP_TRIALS"
  --val-interval "$VAL_INTERVAL"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --early-stopping-min-epoch "$EARLY_STOPPING_MIN_EPOCH"
  --tta-scales "$TTA_SCALES"
  --eval-crop-height "$INPUT_HEIGHT"
  --eval-crop-width "$INPUT_WIDTH"
  --eval-stride-rate 0.6666666667
  --ema-decay 0.9996
  --ema-warmup-epochs 10
  --ce-only-epochs "$CE_ONLY_EPOCHS"
  --ohem-start-epoch "$OHEM_START_EPOCH"
  --ohem-min-kept "$OHEM_MIN_KEPT"
  --lambda-lovasz "$LAMBDA_LOVASZ"
  --lambda-dice "$LAMBDA_DICE"
  --lambda-edge "$LAMBDA_EDGE"
  --lambda-boundary "$LAMBDA_BOUNDARY"
  --lambda-feature-precision "$LAMBDA_FEATURE_PRECISION"
  --feature-precision-max-size "$FEATURE_PRECISION_MAX_SIZE"
  --aux-weight "$AUX_WEIGHT"
  --class-weight-mode "$CLASS_WEIGHT_MODE"
  --class-weight-clamp "$CLASS_WEIGHT_CLAMP"
  --drop-path-rate "$DROP_PATH_RATE"
)
[[ "$ALLOW_CUSTOM_INPUT_SIZE" == "1" ]] && TRAIN_ARGS+=(--allow-custom-input-size)
[[ "$USE_CLASS_WEIGHTS" == "1" ]] && TRAIN_ARGS+=(--use-class-weights)
[[ "$EVAL_NATIVE_SIZE" == "1" ]] && TRAIN_ARGS+=(--eval-native-size)
[[ "$SLIDING_EVAL" == "1" ]] && TRAIN_ARGS+=(--sliding-eval)

EVAL_ARGS=(
  "${DATA_ARGS[@]}"
  "${ARCH_ARGS[@]}"
  --tta-scales "$TTA_SCALES"
  --eval-crop-height "$INPUT_HEIGHT"
  --eval-crop-width "$INPUT_WIDTH"
  --eval-stride-rate 0.6666666667
  --eval-hha-degrade-mode "$EVAL_HHA_DEGRADE_MODE"
  --eval-hha-degrade-severity "$EVAL_HHA_DEGRADE_SEVERITY"
  --eval-hha-degrade-seed "$EVAL_HHA_DEGRADE_SEED"
)
[[ "$ALLOW_CUSTOM_INPUT_SIZE" == "1" ]] && EVAL_ARGS+=(--allow-custom-input-size)
[[ "$EVAL_NATIVE_SIZE" == "1" ]] && EVAL_ARGS+=(--eval-native-size)
[[ "$SLIDING_EVAL" == "1" ]] && EVAL_ARGS+=(--sliding-eval)

TRAIN_LAUNCHER=(python train.py)
EVAL_LAUNCHER=(python evaluate.py)
if [[ "$NUM_GPUS" -gt 1 ]]; then
  TRAIN_LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" train.py)
  EVAL_LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" evaluate.py)
fi

find_best_checkpoint() {
  find "$SAVE_DIR" -maxdepth 1 -type f -name 'best_miou_*.pth' -print 2>/dev/null \
    | sort | tail -n 1
}

case "$MODE" in
  train)
    echo "[RSGNet] train | GPUs=$GPUS | variant=$ARCHITECTURE_VARIANT | source=$GEOMETRY_SOURCE | geometry=$GEOMETRY_ENCODING | fusion=$FUSION_MODE | reliability=$((1 - DISABLE_RELIABILITY))"
    echo "[RSGNet] output=$SAVE_DIR | protocol=$PROTOCOL | seed=$SEED"
    "${TRAIN_LAUNCHER[@]}" "${TRAIN_ARGS[@]}" 2>&1 \
      | tee "$LOG_DIR/train_$TIMESTAMP.log"
    ;;
  resume|finetune)
    if [[ ! -f "$RESUME_CKPT" ]]; then
      echo "[RSGNet] Checkpoint not found: $RESUME_CKPT" >&2
      exit 1
    fi
    EXTRA_ARGS=(--resume "$RESUME_CKPT")
    [[ "$MODE" == "finetune" ]] && EXTRA_ARGS+=(--resume-weights-only)
    "${TRAIN_LAUNCHER[@]}" "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 \
      | tee "$LOG_DIR/${MODE}_$TIMESTAMP.log"
    ;;
  eval)
    default_var VAL_CKPT ""
    [[ -z "$VAL_CKPT" ]] && VAL_CKPT="$(find_best_checkpoint)"
    if [[ -z "$VAL_CKPT" || ! -f "$VAL_CKPT" ]]; then
      echo "[RSGNet] Evaluation checkpoint not found under $SAVE_DIR" >&2
      exit 1
    fi
    EVAL_TTA_ARGS=()
    [[ "$USE_TTA_EVAL" == "1" ]] && EVAL_TTA_ARGS+=(--use-tta)
    "${EVAL_LAUNCHER[@]}" "${EVAL_ARGS[@]}" "${EVAL_TTA_ARGS[@]}" \
      --ckpt "$VAL_CKPT" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
      --metrics-json "$METRICS_JSON" 2>&1 \
      | tee "$LOG_DIR/eval_${EVAL_HHA_DEGRADE_MODE}_${EVAL_HHA_DEGRADE_SEVERITY}_$TIMESTAMP.log"
    ;;
  routes)
    default_var VAL_CKPT ""
    [[ -z "$VAL_CKPT" ]] && VAL_CKPT="$(find_best_checkpoint)"
    if [[ -z "$VAL_CKPT" || ! -f "$VAL_CKPT" ]]; then
      echo "[RSGNet] Route-analysis checkpoint not found under $SAVE_DIR" >&2
      exit 1
    fi
    python tools/analyze_pfhr_routes.py \
      --ckpt "$VAL_CKPT" \
      --max-samples "$ROUTE_MAX_SAMPLES" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --output "$ROUTE_JSON" 2>&1 \
      | tee "$LOG_DIR/routes_$TIMESTAMP.log"
    ;;
  *)
    echo "Usage: bash scripts/run_rsgnet.sh [train|resume|finetune|eval|routes]" >&2
    exit 1
    ;;
esac
