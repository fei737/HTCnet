#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-list}"              # list | train | val | all
EXPERIMENT="${2:-base_simple}"
ABLATION_ROOT="${ABLATION_ROOT:-../ablation_runs}"
SUMMARY_CSV="${SUMMARY_CSV:-$ABLATION_ROOT/ablation_summary.csv}"

mkdir -p "$ABLATION_ROOT"
export GPUS="${GPUS:-2,3}"

define_experiment() {
  local name="$1"

  DECODER_MODE="simple_mlp"
  LAYOUT_MODE="ssm"
  USE_PROMPT_RECOVERY="1"
  PROMPT_RECOVERY_MODE="rgpr"
  USE_CONFIDENCE_ROUTING="1"
  CONSISTENCY_ROUTING_MODE="gacr"
  GEOMETRY_ROUTING_MODE="rcfr"
  FUSION_BRANCH_MODE="both"
  RELIABILITY_STRENGTH="${RELIABILITY_STRENGTH:-0.30}"
  DDP_FIND_UNUSED_PARAMETERS="0"
  SAFE_MODE="0"
  SAVE_DIR="$ABLATION_ROOT/$name"

  case "$name" in
    A0_rgb_baseline)
      PROMPT_RECOVERY_MODE="legacy"
      USE_PROMPT_RECOVERY="0"
      USE_CONFIDENCE_ROUTING="0"
      CONSISTENCY_ROUTING_MODE="none"
      GEOMETRY_ROUTING_MODE="legacy"
      FUSION_BRANCH_MODE="rgb"
      DDP_FIND_UNUSED_PARAMETERS="1"
      ;;
    A1_uncertainty)
      GEOMETRY_ROUTING_MODE="legacy"
      DDP_FIND_UNUSED_PARAMETERS="1"
      ;;
    A2_local_only)
      FUSION_BRANCH_MODE="local"
      DDP_FIND_UNUSED_PARAMETERS="1"
      ;;
    A3_semantic_only)
      FUSION_BRANCH_MODE="semantic"
      DDP_FIND_UNUSED_PARAMETERS="1"
      ;;
    A4_full)
      ;;
    *)
      echo "[ablation] Unknown experiment: $name" >&2
      echo "[ablation] Use: $0 list" >&2
      exit 1
      ;;
  esac
}

experiment_names() {
  cat <<'EOF'
A0_rgb_baseline
A1_uncertainty
A2_local_only
A3_semantic_only
A4_full
EOF
}

print_experiments() {
  printf "%-20s %-12s %-10s %-8s %-8s %-7s %-9s %-9s %-8s\n" "experiment" "decoder" "layout" "routing" "branch" "rec_on" "rec_mode" "confroute" "consist"
  while IFS= read -r name; do
    define_experiment "$name"
    local gsa="on"
    if [[ "$SAFE_MODE" == "1" ]]; then
      gsa="off"
    fi
    local recovery="on"
    if [[ "$USE_PROMPT_RECOVERY" != "1" ]]; then
      recovery="off"
    fi
    local confroute="on"
    if [[ "$USE_CONFIDENCE_ROUTING" != "1" ]]; then
      confroute="off"
    fi
    printf "%-20s %-12s %-10s %-8s %-8s %-7s %-9s %-9s %-8s\n" "$name" "$DECODER_MODE" "$LAYOUT_MODE" "$GEOMETRY_ROUTING_MODE" "$FUSION_BRANCH_MODE" "$recovery" "$PROMPT_RECOVERY_MODE" "$confroute" "$CONSISTENCY_ROUTING_MODE"
  done < <(experiment_names)
}

train_experiment() {
  define_experiment "$1"
  echo "[ablation] train: $1"
  echo "[ablation] save_dir: $SAVE_DIR"
  local train_cmd=(bash run.sh train)
  env \
    SAVE_DIR="$SAVE_DIR" \
    DECODER_MODE="$DECODER_MODE" \
    LAYOUT_MODE="$LAYOUT_MODE" \
    GEOMETRY_ROUTING_MODE="$GEOMETRY_ROUTING_MODE" \
    RELIABILITY_STRENGTH="$RELIABILITY_STRENGTH" \
    FUSION_BRANCH_MODE="$FUSION_BRANCH_MODE" \
    PROMPT_RECOVERY_MODE="$PROMPT_RECOVERY_MODE" \
    CONSISTENCY_ROUTING_MODE="$CONSISTENCY_ROUTING_MODE" \
    USE_PROMPT_RECOVERY="$USE_PROMPT_RECOVERY" \
    USE_CONFIDENCE_ROUTING="$USE_CONFIDENCE_ROUTING" \
    DDP_FIND_UNUSED_PARAMETERS="$DDP_FIND_UNUSED_PARAMETERS" \
    SAFE_MODE="$SAFE_MODE" \
    "${train_cmd[@]}"
}

extract_miou() {
  local log_path="$1"
  python - "$log_path" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, "r", encoding="utf-8", errors="ignore").read()
matches = re.findall(r"Final Validation mIoU:\s*([0-9.]+)", text)
print(matches[-1] if matches else "")
PY
}

ensure_summary_header() {
  local new_header="experiment,decoder,layout,geometry_routing,fusion_branch,prompt_recovery_mode,consistency_routing,gsa,prompt_recovery,confidence_routing,checkpoint,miou"
  local mid_header="experiment,decoder,layout,geometry_routing,gsa,prompt_recovery,confidence_routing,checkpoint,miou"
  local old_header="experiment,decoder,layout,gsa,prompt_recovery,confidence_routing,checkpoint,miou"
  if [[ ! -f "$SUMMARY_CSV" ]]; then
    printf "%s\n" "$new_header" > "$SUMMARY_CSV"
  elif [[ "$(head -n 1 "$SUMMARY_CSV")" == "$mid_header" ]]; then
    local tmp_csv
    tmp_csv="$(mktemp)"
    {
      printf "%s\n" "$new_header"
      tail -n +2 "$SUMMARY_CSV" | awk -F, 'BEGIN{OFS=","} {print $1,$2,$3,$4,"both","legacy","none",$5,$6,$7,$8,$9}'
    } > "$tmp_csv"
    mv "$tmp_csv" "$SUMMARY_CSV"
  elif [[ "$(head -n 1 "$SUMMARY_CSV")" == "$old_header" ]]; then
    local tmp_csv
    tmp_csv="$(mktemp)"
    {
      printf "%s\n" "$new_header"
      tail -n +2 "$SUMMARY_CSV" | awk -F, 'BEGIN{OFS=","} {print $1,$2,$3,"legacy","both","legacy","none",$4,$5,$6,$7,$8}'
    } > "$tmp_csv"
    mv "$tmp_csv" "$SUMMARY_CSV"
  fi
}

append_summary() {
  local name="$1"
  local ckpt="$2"
  local miou="$3"
  local gsa="on"
  local recovery="on"
  local confroute="on"
  if [[ "$SAFE_MODE" == "1" ]]; then
    gsa="off"
  fi
  if [[ "$USE_PROMPT_RECOVERY" != "1" ]]; then
    recovery="off"
  fi
  if [[ "$USE_CONFIDENCE_ROUTING" != "1" ]]; then
    confroute="off"
  fi
  ensure_summary_header
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$name" "$DECODER_MODE" "$LAYOUT_MODE" "$GEOMETRY_ROUTING_MODE" "$FUSION_BRANCH_MODE" "$PROMPT_RECOVERY_MODE" "$CONSISTENCY_ROUTING_MODE" "$gsa" "$recovery" "$confroute" "$ckpt" "$miou" \
    >> "$SUMMARY_CSV"
}

validate_experiment() {
  define_experiment "$1"
  local best_ckpt
  best_ckpt="$(find "$SAVE_DIR" -maxdepth 1 -type f -name 'best_miou_*.pth' | sort | tail -n 1 || true)"
  if [[ -z "$best_ckpt" ]]; then
    echo "[ablation] no best checkpoint found in $SAVE_DIR" >&2
    exit 1
  fi

  local safe_args=()
  if [[ "$SAFE_MODE" == "1" ]]; then
    safe_args+=(--safe-mode)
  fi

  local val_log="$SAVE_DIR/ablation_val_$(date +%Y%m%d_%H%M%S).log"
  echo "[ablation] val: $1"
  echo "[ablation] checkpoint: $best_ckpt"
  if [[ -n "${GPUS:-}" ]]; then
    echo "[ablation] val CUDA_VISIBLE_DEVICES: $GPUS"
  fi

  local val_cmd=(python val.py \
	    --data-root "${DATA_ROOT:-/home/pengfei/HTCnet/DataSets}" \
	    --dataset-name "${DATASET_NAME:-SUNRGBD}" \
	    --ckpt "$best_ckpt" \
    --batch-size "${BATCH_SIZE:-4}" \
    --num-workers "${NUM_WORKERS:-8}" \
    --n-classes "${N_CLASSES:-41}" \
	    --input-height "${INPUT_HEIGHT:-480}" \
	    --input-width "${INPUT_WIDTH:-640}" \
	    --label-dir-name "${LABEL_DIR_NAME:-Labels}" \
	    --hha-name-prefix "${HHA_NAME_PREFIX:-}" \
	    --label-name-prefix "${LABEL_NAME_PREFIX:-}" \
	    --ignore-index "${IGNORE_INDEX:-0}" \
    --miou-start-class "${MIOU_START_CLASS:-1}" \
    --encoder-name "${ENCODER_NAME:-mit_b2}" \
    --use-tta \
    --prompt-channels "${PROMPT_CHANNELS:-48}" \
    --layout-state-dim "${LAYOUT_STATE_DIM:-24}" \
	    --geometry-routing-mode "$GEOMETRY_ROUTING_MODE" \
	    --reliability-strength "$RELIABILITY_STRENGTH" \
	    --fusion-branch-mode "$FUSION_BRANCH_MODE" \
    --prompt-recovery-mode "$PROMPT_RECOVERY_MODE" \
    --consistency-routing-mode "$CONSISTENCY_ROUTING_MODE" \
    --decoder-channels "${DECODER_CHANNELS:-320}" \
    --decoder-mode "$DECODER_MODE" \
    --attention-tokens "${ATTENTION_TOKENS:-1600}" \
    --local-scale-init "${LOCAL_SCALE_INIT:-0.001}" \
    --semantic-scale-init "${SEMANTIC_SCALE_INIT:-0.001}" \
    --query-points "${QUERY_POINTS:-6}" \
    --query-scale-init "${QUERY_SCALE_INIT:-0.08}" \
    --geometry-scale-init "${GEOMETRY_SCALE_INIT:-0.15}" \
    --detail-logit-scale "${DETAIL_LOGIT_SCALE:-0.35}" \
    --rgb-boundary-scale "${RGB_BOUNDARY_SCALE:-0.15}" \
    --layout-mode "$LAYOUT_MODE")
  if [[ "$USE_PROMPT_RECOVERY" != "1" ]]; then
    val_cmd+=(--no-prompt-recovery)
  fi
  if [[ "$USE_CONFIDENCE_ROUTING" != "1" ]]; then
    val_cmd+=(--no-confidence-routing)
  fi
  val_cmd+=("${safe_args[@]}")

  if [[ -n "${GPUS:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$GPUS" "${val_cmd[@]}" 2>&1 | tee "$val_log"
  else
    "${val_cmd[@]}" 2>&1 | tee "$val_log"
  fi

  local miou
  miou="$(extract_miou "$val_log")"
  append_summary "$1" "$best_ckpt" "$miou"
  echo "[ablation] summary updated: $SUMMARY_CSV"
  if [[ -n "$miou" ]]; then
    echo "[ablation] final miou: $miou"
  fi
}

case "$MODE" in
  list)
    print_experiments
    ;;
  train)
    train_experiment "$EXPERIMENT"
    ;;
  val)
    validate_experiment "$EXPERIMENT"
    ;;
  all)
    train_experiment "$EXPERIMENT"
    validate_experiment "$EXPERIMENT"
    ;;
  *)
    echo "Usage: bash ablation.sh [list|train|val|all] [experiment_name]"
    exit 1
    ;;
esac
