#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

MODE="${1:-train}"              # train | resume | finetune | val | vis
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# --- Paths --------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/home/pengfei/HTCnet/DataSets}"
DATASET_NAME="${DATASET_NAME:-SUNRGBD}"              # NYU | SUNRGBD
DECODER_MODE="${DECODER_MODE:-simple_mlp}"       # simple_mlp (decoder ablation) | qdhs
SAVE_DIR="${SAVE_DIR:-../Checkpoint_${DATASET_NAME}_${DECODER_MODE}}"
PRETRAINED_ENCODER="${PRETRAINED_ENCODER:-/home/pengfei/HTCnet/Checkpoint/mit_b2.pth}"
RESUME_CKPT="${RESUME_CKPT:-$SAVE_DIR/latest_model.pth}"
BEST_CKPT="$(find "$SAVE_DIR" -maxdepth 1 -type f -name 'best_miou_*.pth' 2>/dev/null | sort | tail -n 1 || true)"
VAL_CKPT="${VAL_CKPT:-${BEST_CKPT:-$SAVE_DIR/best_miou_UNKNOWN.pth}}"

# --- Common knobs -------------------------------------------------------------
GPUS="${GPUS:-2,3}"                       # comma-separated CUDA device ids
ENCODER_NAME="${ENCODER_NAME:-mit_b2}"
LAYOUT_MODE="${LAYOUT_MODE:-ssm}"         # ssm (proposed) | conv | avgpool (ablation)
N_CLASSES="${N_CLASSES:-41}"
INPUT_HEIGHT="${INPUT_HEIGHT:-480}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"
BATCH_SIZE="${BATCH_SIZE:-4}"             # per-GPU batch
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-8e-5}"
ENCODER_LR_MULT="${ENCODER_LR_MULT:-0.5}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"            # bf16 | fp16
VAL_INTERVAL="${VAL_INTERVAL:-1}"
if [[ -z "${LABEL_DIR_NAME+x}" ]]; then
  if [[ "$DATASET_NAME" == "NYU" ]]; then
    LABEL_DIR_NAME="Label"
  else
    LABEL_DIR_NAME="Labels"
  fi
fi
if [[ -z "${HHA_NAME_PREFIX+x}" ]]; then
  if [[ "$DATASET_NAME" == "NYU" ]]; then
    HHA_NAME_PREFIX="hha_"
  else
    HHA_NAME_PREFIX=""
  fi
fi
LABEL_NAME_PREFIX="${LABEL_NAME_PREFIX:-}"
IGNORE_INDEX="${IGNORE_INDEX:-0}"
MIOU_START_CLASS="${MIOU_START_CLASS:-1}"
TRAIN_CUTOUT_PROB="${TRAIN_CUTOUT_PROB:-0.10}"
USE_CLASS_WEIGHTS="${USE_CLASS_WEIGHTS:-1}"        # 1 to enable inverse-frequency class weights
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-inverse_log}"  # inverse_log | inverse_freq
CLASS_WEIGHT_CLAMP="${CLASS_WEIGHT_CLAMP:-4.0}"   # lower spread is steadier with OHEM
AUG_LEVEL="${AUG_LEVEL:-base}"                    # base | strong
DROP_PATH_RATE="${DROP_PATH_RATE:-0.05}"          # encoder stochastic depth
USE_GRAD_CHECKPOINT="${USE_GRAD_CHECKPOINT:-1}"   # 1: recompute attention in backward to save memory (fixes OOM)
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-0}" # 1: needed for ablations that bypass whole branches
SAFE_MODE="${SAFE_MODE:-0}"                       # 1: disable deep angular global attention
USE_PROMPT_RECOVERY="${USE_PROMPT_RECOVERY:-1}"   # bounded HHA recovery before geometry prompts
PROMPT_RECOVERY_MODE="${PROMPT_RECOVERY_MODE:-rgpr}" # legacy | rgpr (RGB-guided prompt recovery)
USE_CONFIDENCE_ROUTING="${USE_CONFIDENCE_ROUTING:-1}" # suppress global geometry when HHA is unreliable
CONSISTENCY_ROUTING_MODE="${CONSISTENCY_ROUTING_MODE:-gacr}" # none | gacr (geometry-appearance consistency)
PROMPT_CHANNELS="${PROMPT_CHANNELS:-48}"          # geometry prompt width; 32 is V7, 48 is V9 capacity run
LAYOUT_STATE_DIM="${LAYOUT_STATE_DIM:-24}"        # existing SSM state width; 16 is V7
GEOMETRY_ROUTING_MODE="${GEOMETRY_ROUTING_MODE:-rcfr}" # legacy | rcfr (reliability-calibrated frequency routing)
RELIABILITY_STRENGTH="${RELIABILITY_STRENGTH:-0.30}"
FUSION_BRANCH_MODE="${FUSION_BRANCH_MODE:-both}" # both | local | semantic | rgb
USE_PROMPT_AUTOCORR="${USE_PROMPT_AUTOCORR:-1}" # AFFN-inspired reliability-aware prompt autocorrelation
USE_DIRECTIONAL_EDGE_REFINE="${USE_DIRECTIONAL_EDGE_REFINE:-1}" # DEGConv-inspired directional geometry edge refinement
DECODER_CHANNELS="${DECODER_CHANNELS:-320}"       # existing decoder width; 256 is V7
ATTENTION_TOKENS="${ATTENTION_TOKENS:-1600}"      # existing semantic attention budget; 1024 is V7
LOCAL_SCALE_INIT="${LOCAL_SCALE_INIT:-0.001}"     # existing local residual starts less muted
SEMANTIC_SCALE_INIT="${SEMANTIC_SCALE_INIT:-0.001}" # existing semantic residual starts less muted
QUERY_POINTS="${QUERY_POINTS:-6}"                 # existing query attention samples; 4 is V7
QUERY_SCALE_INIT="${QUERY_SCALE_INIT:-0.08}"      # existing query mask residual scale
GEOMETRY_SCALE_INIT="${GEOMETRY_SCALE_INIT:-0.15}" # existing geometry reconstruction scale
DETAIL_LOGIT_SCALE="${DETAIL_LOGIT_SCALE:-0.35}"  # existing detail-logit residual
RGB_BOUNDARY_SCALE="${RGB_BOUNDARY_SCALE:-0.15}"  # existing RGB boundary residual
AUTO_LR_START_EPOCH="${AUTO_LR_START_EPOCH:-75}"
AUTO_LR_PATIENCE="${AUTO_LR_PATIENCE:-10}"
CE_ONLY_EPOCHS="${CE_ONLY_EPOCHS:-10}"
OHEM_START_EPOCH="${OHEM_START_EPOCH:-40}"
OHEM_MIN_KEPT="${OHEM_MIN_KEPT:-80000}"
FEATURE_PRECISION_MAX_SIZE="${FEATURE_PRECISION_MAX_SIZE:-160}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# --- Dataset sanity check (skipped for val/vis single-image use) --------------
DATASET_ROOT="$DATA_ROOT/$DATASET_NAME"
for path in "$DATASET_ROOT/RGB" "$DATASET_ROOT/HHA" "$DATASET_ROOT/$LABEL_DIR_NAME" "$DATASET_ROOT/train.txt" "$DATASET_ROOT/test.txt"; do
  if [[ ! -e "$path" ]]; then
    echo "[run] ERROR: missing $DATASET_NAME path: $path"
    echo "[run] Expected: \$DATA_ROOT/$DATASET_NAME/{RGB,HHA,$LABEL_DIR_NAME,train.txt,test.txt}"
    exit 1
  fi
done

# --- GPU count (for DDP launch) ----------------------------------------------
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_GPUS="${#GPU_ARRAY[@]}"
if command -v python >/dev/null 2>&1; then
  VISIBLE_NUM_GPUS="$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
)"
  NUM_GPUS="$VISIBLE_NUM_GPUS"
fi

# --- Training recipe (proven defaults; the per-GPU/global batch split and BN
#     sync are handled inside train.py so mIoU matches across GPU counts) ------
COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --dataset-name "$DATASET_NAME"
  --save-dir "$SAVE_DIR"
  --pretrained-encoder "$PRETRAINED_ENCODER"
  --encoder-name "$ENCODER_NAME"
  --layout-mode "$LAYOUT_MODE"
  --decoder-mode "$DECODER_MODE"
  --n-classes "$N_CLASSES"
  --ignore-index "$IGNORE_INDEX"
  --miou-start-class "$MIOU_START_CLASS"
  --input-height "$INPUT_HEIGHT"
  --input-width "$INPUT_WIDTH"
  --batch-size "$BATCH_SIZE"
  --target-global-batch "$TARGET_GLOBAL_BATCH"
  --num-workers "$NUM_WORKERS"
  --epochs "$EPOCHS"
  --lr "$LR"
  --encoder-lr-mult "$ENCODER_LR_MULT"
  --warmup-epochs 5
  --min-lr-ratio "$MIN_LR_RATIO"
  --amp-dtype "$AMP_DTYPE"
  --label-dir-name "$LABEL_DIR_NAME"
  --hha-name-prefix "$HHA_NAME_PREFIX"
  --label-name-prefix "$LABEL_NAME_PREFIX"
  --train-cutout-prob "$TRAIN_CUTOUT_PROB"
  --val-interval "$VAL_INTERVAL"
  --ema-decay 0.9996
  --ema-warmup-epochs 10
  --ce-only-epochs "$CE_ONLY_EPOCHS"
  --ohem-start-epoch "$OHEM_START_EPOCH"
  --ohem-min-kept "$OHEM_MIN_KEPT"
  --lambda-lovasz 0.5
  --lambda-dice 0.4
  --lambda-edge 0.03
  --lambda-boundary 0.01
  --lambda-feature-precision 0.01
  --feature-precision-max-size "$FEATURE_PRECISION_MAX_SIZE"
  --aux-weight 0.1
  --class-weight-mode "$CLASS_WEIGHT_MODE"
  --class-weight-clamp "$CLASS_WEIGHT_CLAMP"
  --aug-level "$AUG_LEVEL"
  --drop-path-rate "$DROP_PATH_RATE"
  --prompt-channels "$PROMPT_CHANNELS"
  --layout-state-dim "$LAYOUT_STATE_DIM"
  --geometry-routing-mode "$GEOMETRY_ROUTING_MODE"
  --reliability-strength "$RELIABILITY_STRENGTH"
  --prompt-recovery-mode "$PROMPT_RECOVERY_MODE"
  --consistency-routing-mode "$CONSISTENCY_ROUTING_MODE"
  --fusion-branch-mode "$FUSION_BRANCH_MODE"
  --decoder-channels "$DECODER_CHANNELS"
  --attention-tokens "$ATTENTION_TOKENS"
  --local-scale-init "$LOCAL_SCALE_INIT"
  --semantic-scale-init "$SEMANTIC_SCALE_INIT"
  --query-points "$QUERY_POINTS"
  --query-scale-init "$QUERY_SCALE_INIT"
  --geometry-scale-init "$GEOMETRY_SCALE_INIT"
  --detail-logit-scale "$DETAIL_LOGIT_SCALE"
  --rgb-boundary-scale "$RGB_BOUNDARY_SCALE"
  --auto-lr
  --auto-lr-start-epoch "$AUTO_LR_START_EPOCH"
  --auto-lr-patience "$AUTO_LR_PATIENCE"
  --auto-lr-factor 0.7
  --auto-lr-max-reductions 3
)

if [[ "$USE_CLASS_WEIGHTS" == "1" ]]; then
  COMMON_ARGS+=(--use-class-weights)
fi
if [[ "$USE_GRAD_CHECKPOINT" == "1" ]]; then
  COMMON_ARGS+=(--use-grad-checkpoint)
fi
if [[ "$DDP_FIND_UNUSED_PARAMETERS" == "1" ]]; then
  COMMON_ARGS+=(--ddp-find-unused-parameters)
fi
if [[ "$SAFE_MODE" == "1" ]]; then
  COMMON_ARGS+=(--safe-mode)
fi
if [[ "$USE_PROMPT_RECOVERY" != "1" ]]; then
  COMMON_ARGS+=(--no-prompt-recovery)
fi
if [[ "$USE_CONFIDENCE_ROUTING" != "1" ]]; then
  COMMON_ARGS+=(--no-confidence-routing)
fi
if [[ "$USE_PROMPT_AUTOCORR" != "1" ]]; then
  COMMON_ARGS+=(--no-prompt-autocorr)
fi
if [[ "$USE_DIRECTIONAL_EDGE_REFINE" != "1" ]]; then
  COMMON_ARGS+=(--no-directional-edge-refine)
fi

VAL_ARGS=(
  --data-root "$DATA_ROOT"
  --dataset-name "$DATASET_NAME"
  --n-classes "$N_CLASSES"
  --ignore-index "$IGNORE_INDEX"
  --miou-start-class "$MIOU_START_CLASS"
  --input-height "$INPUT_HEIGHT"
  --input-width "$INPUT_WIDTH"
  --label-dir-name "$LABEL_DIR_NAME"
  --hha-name-prefix "$HHA_NAME_PREFIX"
  --label-name-prefix "$LABEL_NAME_PREFIX"
  --encoder-name "$ENCODER_NAME"
  --layout-mode "$LAYOUT_MODE"
  --decoder-mode "$DECODER_MODE"
  --prompt-channels "$PROMPT_CHANNELS"
  --layout-state-dim "$LAYOUT_STATE_DIM"
  --geometry-routing-mode "$GEOMETRY_ROUTING_MODE"
  --reliability-strength "$RELIABILITY_STRENGTH"
  --prompt-recovery-mode "$PROMPT_RECOVERY_MODE"
  --consistency-routing-mode "$CONSISTENCY_ROUTING_MODE"
  --fusion-branch-mode "$FUSION_BRANCH_MODE"
  --decoder-channels "$DECODER_CHANNELS"
  --attention-tokens "$ATTENTION_TOKENS"
  --local-scale-init "$LOCAL_SCALE_INIT"
  --semantic-scale-init "$SEMANTIC_SCALE_INIT"
  --query-points "$QUERY_POINTS"
  --query-scale-init "$QUERY_SCALE_INIT"
  --geometry-scale-init "$GEOMETRY_SCALE_INIT"
  --detail-logit-scale "$DETAIL_LOGIT_SCALE"
  --rgb-boundary-scale "$RGB_BOUNDARY_SCALE"
)
if [[ "$USE_PROMPT_RECOVERY" != "1" ]]; then
  VAL_ARGS+=(--no-prompt-recovery)
fi
if [[ "$USE_CONFIDENCE_ROUTING" != "1" ]]; then
  VAL_ARGS+=(--no-confidence-routing)
fi
if [[ "$USE_PROMPT_AUTOCORR" != "1" ]]; then
  VAL_ARGS+=(--no-prompt-autocorr)
fi
if [[ "$USE_DIRECTIONAL_EDGE_REFINE" != "1" ]]; then
  VAL_ARGS+=(--no-directional-edge-refine)
fi
if [[ "$SAFE_MODE" == "1" ]]; then
  VAL_ARGS+=(--safe-mode)
fi

TRAIN_LAUNCHER=(python train.py)
if [[ "$NUM_GPUS" -gt 1 ]]; then
  TRAIN_LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" train.py)
fi

case "$MODE" in
  train)
    echo "[run] Training UGF-Lite on GPUs: $CUDA_VISIBLE_DEVICES (procs=$NUM_GPUS)"
    echo "[run] Dataset: $DATASET_ROOT | labels: $LABEL_DIR_NAME | hha_prefix: '$HHA_NAME_PREFIX' | classes: $N_CLASSES | ignore: $IGNORE_INDEX"
    echo "[run] Save dir: $SAVE_DIR | per-GPU batch: $BATCH_SIZE | global batch: $TARGET_GLOBAL_BATCH"
    echo "[run] LR: $LR | encoder_lr_mult=$ENCODER_LR_MULT | min_lr_ratio=$MIN_LR_RATIO | epochs=$EPOCHS | layout=$LAYOUT_MODE | amp=$AMP_DTYPE"
    echo "[run] decoder: $DECODER_MODE"
    echo "[run] recipe: ce_only=$CE_ONLY_EPOCHS ohem_start=$OHEM_START_EPOCH ohem_min_kept=$OHEM_MIN_KEPT cutout_prob=$TRAIN_CUTOUT_PROB aug=$AUG_LEVEL"
    echo "[run] capacity: prompt=$PROMPT_CHANNELS state=$LAYOUT_STATE_DIM decoder=$DECODER_CHANNELS attn_tokens=$ATTENTION_TOKENS query_points=$QUERY_POINTS"
    echo "[run] geometry routing: mode=$GEOMETRY_ROUTING_MODE reliability_strength=$RELIABILITY_STRENGTH"
    echo "[run] geometry controls: prompt_recovery=$USE_PROMPT_RECOVERY prompt_recovery_mode=$PROMPT_RECOVERY_MODE confidence_routing=$USE_CONFIDENCE_ROUTING prompt_autocorr=$USE_PROMPT_AUTOCORR directional_edge=$USE_DIRECTIONAL_EDGE_REFINE consistency=$CONSISTENCY_ROUTING_MODE fusion_branch=$FUSION_BRANCH_MODE"
    echo "[run] DDP: find_unused_parameters=$DDP_FIND_UNUSED_PARAMETERS"
    "${TRAIN_LAUNCHER[@]}" "${COMMON_ARGS[@]}" 2>&1 | tee "logs/train_${TIMESTAMP}.log"
    ;;

  resume)
    echo "[run] Resuming UGF-Lite from: $RESUME_CKPT (procs=$NUM_GPUS)"
    echo "[run] Dataset: $DATASET_ROOT | labels: $LABEL_DIR_NAME | hha_prefix: '$HHA_NAME_PREFIX' | classes: $N_CLASSES | ignore: $IGNORE_INDEX"
    echo "[run] LR: $LR | encoder_lr_mult=$ENCODER_LR_MULT | min_lr_ratio=$MIN_LR_RATIO | cutout_prob=$TRAIN_CUTOUT_PROB | layout=$LAYOUT_MODE"
    echo "[run] decoder: $DECODER_MODE"
    echo "[run] capacity: prompt=$PROMPT_CHANNELS state=$LAYOUT_STATE_DIM decoder=$DECODER_CHANNELS attn_tokens=$ATTENTION_TOKENS query_points=$QUERY_POINTS"
    echo "[run] geometry routing: mode=$GEOMETRY_ROUTING_MODE reliability_strength=$RELIABILITY_STRENGTH"
    echo "[run] geometry controls: prompt_recovery=$USE_PROMPT_RECOVERY prompt_recovery_mode=$PROMPT_RECOVERY_MODE confidence_routing=$USE_CONFIDENCE_ROUTING prompt_autocorr=$USE_PROMPT_AUTOCORR directional_edge=$USE_DIRECTIONAL_EDGE_REFINE consistency=$CONSISTENCY_ROUTING_MODE fusion_branch=$FUSION_BRANCH_MODE"
    echo "[run] DDP: find_unused_parameters=$DDP_FIND_UNUSED_PARAMETERS"
    "${TRAIN_LAUNCHER[@]}" "${COMMON_ARGS[@]}" --resume "$RESUME_CKPT" 2>&1 | tee "logs/resume_${TIMESTAMP}.log"
    ;;

  finetune)
    if [[ ! -f "$RESUME_CKPT" ]]; then
      echo "[run] ERROR: fine-tune checkpoint not found: $RESUME_CKPT"
      echo "[run] Set RESUME_CKPT=/path/to/checkpoint.pth"
      exit 1
    fi
    echo "[run] Fine-tuning UGF-Lite weights from: $RESUME_CKPT (procs=$NUM_GPUS)"
    echo "[run] Dataset: $DATASET_ROOT | labels: $LABEL_DIR_NAME | hha_prefix: '$HHA_NAME_PREFIX' | classes: $N_CLASSES | ignore: $IGNORE_INDEX"
    echo "[run] Optimizer/scheduler will be reset | LR: $LR | encoder_lr_mult=$ENCODER_LR_MULT | layout=$LAYOUT_MODE | drop_path=$DROP_PATH_RATE"
    echo "[run] decoder: $DECODER_MODE"
    echo "[run] capacity: prompt=$PROMPT_CHANNELS state=$LAYOUT_STATE_DIM decoder=$DECODER_CHANNELS attn_tokens=$ATTENTION_TOKENS query_points=$QUERY_POINTS"
    echo "[run] geometry routing: mode=$GEOMETRY_ROUTING_MODE reliability_strength=$RELIABILITY_STRENGTH"
    echo "[run] geometry controls: prompt_recovery=$USE_PROMPT_RECOVERY prompt_recovery_mode=$PROMPT_RECOVERY_MODE confidence_routing=$USE_CONFIDENCE_ROUTING prompt_autocorr=$USE_PROMPT_AUTOCORR directional_edge=$USE_DIRECTIONAL_EDGE_REFINE consistency=$CONSISTENCY_ROUTING_MODE fusion_branch=$FUSION_BRANCH_MODE"
    echo "[run] DDP: find_unused_parameters=$DDP_FIND_UNUSED_PARAMETERS"
    "${TRAIN_LAUNCHER[@]}" "${COMMON_ARGS[@]}" --resume "$RESUME_CKPT" --resume-weights-only 2>&1 | tee "logs/finetune_${TIMESTAMP}.log"
    ;;

  val)
    if [[ ! -f "$VAL_CKPT" ]]; then
      echo "[run] ERROR: validation checkpoint not found: $VAL_CKPT"
      echo "[run] Set VAL_CKPT=/path/to/checkpoint.pth"
      exit 1
    fi
    echo "[run] Validating UGF-Lite checkpoint: $VAL_CKPT"
    echo "[run] Dataset: $DATASET_ROOT | labels: $LABEL_DIR_NAME | hha_prefix: '$HHA_NAME_PREFIX' | classes: $N_CLASSES | ignore: $IGNORE_INDEX"
    python val.py "${VAL_ARGS[@]}" \
      --ckpt "$VAL_CKPT" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --use-tta \
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
    shift 2  # drop mode + image name; remaining args pass through to inference.py
    if [[ ! -f "$VAL_CKPT" ]]; then
      echo "[run] ERROR: visualization checkpoint not found: $VAL_CKPT"
      echo "[run] Set VAL_CKPT=/path/to/checkpoint.pth"
      exit 1
    fi
    echo "[run] Visualizing $IMG_NAME with checkpoint: $VAL_CKPT"
    VIS_GEOMETRY_ARGS=()
    if [[ "$USE_PROMPT_RECOVERY" != "1" ]]; then
      VIS_GEOMETRY_ARGS+=(--no-prompt-recovery)
    fi
    if [[ "$USE_CONFIDENCE_ROUTING" != "1" ]]; then
      VIS_GEOMETRY_ARGS+=(--no-confidence-routing)
    fi
    if [[ "$USE_PROMPT_AUTOCORR" != "1" ]]; then
      VIS_GEOMETRY_ARGS+=(--no-prompt-autocorr)
    fi
    if [[ "$USE_DIRECTIONAL_EDGE_REFINE" != "1" ]]; then
      VIS_GEOMETRY_ARGS+=(--no-directional-edge-refine)
    fi
    python inference.py \
      --rgb "$DATASET_ROOT/RGB/$IMG_NAME" \
      --hha "$DATASET_ROOT/HHA/${HHA_NAME_PREFIX}${BASE_NAME}.png" \
      --label "$DATASET_ROOT/$LABEL_DIR_NAME/${LABEL_NAME_PREFIX}${BASE_NAME}.png" \
      --ckpt "$VAL_CKPT" \
      --n-classes "$N_CLASSES" \
      --input-height "$INPUT_HEIGHT" \
      --input-width "$INPUT_WIDTH" \
      --encoder-name "$ENCODER_NAME" \
      --layout-mode "$LAYOUT_MODE" \
      --decoder-mode "$DECODER_MODE" \
      --prompt-channels "$PROMPT_CHANNELS" \
      --layout-state-dim "$LAYOUT_STATE_DIM" \
      --geometry-routing-mode "$GEOMETRY_ROUTING_MODE" \
      --reliability-strength "$RELIABILITY_STRENGTH" \
      --prompt-recovery-mode "$PROMPT_RECOVERY_MODE" \
      --consistency-routing-mode "$CONSISTENCY_ROUTING_MODE" \
      --fusion-branch-mode "$FUSION_BRANCH_MODE" \
      --decoder-channels "$DECODER_CHANNELS" \
      --attention-tokens "$ATTENTION_TOKENS" \
      --local-scale-init "$LOCAL_SCALE_INIT" \
      --semantic-scale-init "$SEMANTIC_SCALE_INIT" \
      --query-points "$QUERY_POINTS" \
      --query-scale-init "$QUERY_SCALE_INIT" \
      --geometry-scale-init "$GEOMETRY_SCALE_INIT" \
      --detail-logit-scale "$DETAIL_LOGIT_SCALE" \
      --rgb-boundary-scale "$RGB_BOUNDARY_SCALE" \
      "${VIS_GEOMETRY_ARGS[@]}" \
      --save-path "logs/vis_${BASE_NAME}_${TIMESTAMP}.png" \
      "$@" \
      2>&1 | tee "logs/vis_${BASE_NAME}_${TIMESTAMP}.log"
    ;;

  *)
    echo "Usage: bash run.sh [train|resume|finetune|val|vis]"
    exit 1
    ;;
esac
