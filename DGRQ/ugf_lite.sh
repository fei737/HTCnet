#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-list}"                 # list | train | val | all | all-phase1 | auto-core | auto-all | report
EXPERIMENT="${2:-E5_full_lite}"
UGF_ROOT="${UGF_ROOT:-../runs_ugf_lite}"
SUMMARY_CSV="${SUMMARY_CSV:-$UGF_ROOT/ugf_lite_summary.csv}"
SUMMARY_HEADER="experiment,phase,dataset,decoder,layout,fusion_branch,prompt_recovery,prompt_autocorr,directional_edge,confidence_routing,consistency_routing,safe_mode,geometry_routing,prompt_channels,local_scale,semantic_scale,encoder_lr_mult,drop_path,checkpoint,eval_policy,miou"
mkdir -p "$UGF_ROOT"

export GPUS="${GPUS:-2,3}"

BASE_LAYOUT_MODE="${UGF_LAYOUT_MODE:-conv}"
BASE_RELIABILITY_STRENGTH="${RELIABILITY_STRENGTH:-0.30}"
BASE_PROMPT_CHANNELS="${PROMPT_CHANNELS:-48}"
BASE_LAYOUT_STATE_DIM="${LAYOUT_STATE_DIM:-24}"
BASE_DECODER_CHANNELS="${DECODER_CHANNELS:-320}"
BASE_ATTENTION_TOKENS="${ATTENTION_TOKENS:-1600}"
BASE_LOCAL_SCALE_INIT="${LOCAL_SCALE_INIT:-0.001}"
BASE_SEMANTIC_SCALE_INIT="${SEMANTIC_SCALE_INIT:-0.001}"
BASE_USE_PROMPT_AUTOCORR="${USE_PROMPT_AUTOCORR:-0}"
BASE_USE_DIRECTIONAL_EDGE_REFINE="${USE_DIRECTIONAL_EDGE_REFINE:-0}"
BASE_ENCODER_LR_MULT="${ENCODER_LR_MULT:-0.5}"
BASE_DROP_PATH_RATE="${DROP_PATH_RATE:-0.05}"
BASE_EPOCHS="${EPOCHS:-100}"

# This script keeps the paper-facing model line clean:
# strong RGB encoder + reliability-aware geometry prompt/fusion + simple decoder.
# It intentionally disables QDHS, RCFR, GSA, and GACR unless an experiment asks
# for them, so mIoU changes can be attributed to encoder/fusion design.

set_common_lite_defaults() {
  DECODER_MODE="simple_mlp"
  LAYOUT_MODE="$BASE_LAYOUT_MODE"
  GEOMETRY_ROUTING_MODE="legacy"
  RELIABILITY_STRENGTH="$BASE_RELIABILITY_STRENGTH"
  FUSION_BRANCH_MODE="both"
  USE_PROMPT_RECOVERY="1"
  PROMPT_RECOVERY_MODE="rgpr"
  USE_CONFIDENCE_ROUTING="0"
  CONSISTENCY_ROUTING_MODE="none"
  SAFE_MODE="1"
  DDP_FIND_UNUSED_PARAMETERS="0"

  PROMPT_CHANNELS="$BASE_PROMPT_CHANNELS"
  LAYOUT_STATE_DIM="$BASE_LAYOUT_STATE_DIM"
  DECODER_CHANNELS="$BASE_DECODER_CHANNELS"
  ATTENTION_TOKENS="$BASE_ATTENTION_TOKENS"
  LOCAL_SCALE_INIT="$BASE_LOCAL_SCALE_INIT"
  SEMANTIC_SCALE_INIT="$BASE_SEMANTIC_SCALE_INIT"
  USE_PROMPT_AUTOCORR="$BASE_USE_PROMPT_AUTOCORR"
  USE_DIRECTIONAL_EDGE_REFINE="$BASE_USE_DIRECTIONAL_EDGE_REFINE"
  ENCODER_LR_MULT="$BASE_ENCODER_LR_MULT"
  DROP_PATH_RATE="$BASE_DROP_PATH_RATE"
  EPOCHS="$BASE_EPOCHS"
  SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
  PHASE="phase1_main_ablation"
  HYPOTHESIS="clean encoder/fusion ablation"
}

define_experiment() {
  EXPERIMENT="$1"
  set_common_lite_defaults
  SAVE_DIR="$UGF_ROOT/$EXPERIMENT"

  case "$EXPERIMENT" in
    E0_rgb_only)
      PHASE="phase1_main_ablation"
      HYPOTHESIS="RGB encoder + simple decoder baseline; geometry must not affect logits."
      FUSION_BRANCH_MODE="rgb"
      USE_PROMPT_RECOVERY="0"
      USE_CONFIDENCE_ROUTING="0"
      CONSISTENCY_ROUTING_MODE="none"
      GEOMETRY_ROUTING_MODE="legacy"
      DDP_FIND_UNUSED_PARAMETERS="1"
      ;;
    E1_hha_prompt)
      PHASE="phase1_main_ablation"
      HYPOTHESIS="Naive HHA-derived geometry prompt without recovery or confidence routing."
      USE_PROMPT_RECOVERY="0"
      USE_CONFIDENCE_ROUTING="0"
      CONSISTENCY_ROUTING_MODE="none"
      GEOMETRY_ROUTING_MODE="legacy"
      FUSION_BRANCH_MODE="both"
      DDP_FIND_UNUSED_PARAMETERS="1"
      ;;
    E2_rgpr)
      PHASE="phase1_main_ablation"
      HYPOTHESIS="RGB-guided bounded HHA recovery should improve geometry prompt quality."
      USE_PROMPT_RECOVERY="1"
      PROMPT_RECOVERY_MODE="rgpr"
      USE_CONFIDENCE_ROUTING="0"
      CONSISTENCY_ROUTING_MODE="none"
      GEOMETRY_ROUTING_MODE="legacy"
      FUSION_BRANCH_MODE="both"
      ;;
    E3_local_only)
      PHASE="phase1_main_ablation"
      HYPOTHESIS="Local alignment branch contribution under the same recovered geometry."
      FUSION_BRANCH_MODE="local"
      DDP_FIND_UNUSED_PARAMETERS="1"
      ;;
    E4_semantic_only)
      PHASE="phase1_main_ablation"
      HYPOTHESIS="Semantic/global branch contribution under the same recovered geometry."
      FUSION_BRANCH_MODE="semantic"
      DDP_FIND_UNUSED_PARAMETERS="1"
      ;;
    E5_full_lite)
      PHASE="phase1_main_ablation"
      HYPOTHESIS="Full UGF-Lite encoder/fusion line: recovered geometry + directional edge refinement + autocorrelation prompt mixing + local and semantic branches."
      FUSION_BRANCH_MODE="both"
      USE_PROMPT_AUTOCORR="1"
      USE_DIRECTIONAL_EDGE_REFINE="1"
      ;;

    R_003_003)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="phase2_residual_scale"
      HYPOTHESIS="Residual scale sweep: balanced 0.003/0.003."
      LOCAL_SCALE_INIT="0.003"
      SEMANTIC_SCALE_INIT="0.003"
      ;;
    R_005_003)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="phase2_residual_scale"
      HYPOTHESIS="Residual scale sweep: stronger local branch."
      LOCAL_SCALE_INIT="0.005"
      SEMANTIC_SCALE_INIT="0.003"
      ;;
    R_003_005)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="phase2_residual_scale"
      HYPOTHESIS="Residual scale sweep: stronger semantic branch."
      LOCAL_SCALE_INIT="0.003"
      SEMANTIC_SCALE_INIT="0.005"
      ;;
    R_005_005)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="phase2_residual_scale"
      HYPOTHESIS="Residual scale sweep: balanced 0.005/0.005."
      LOCAL_SCALE_INIT="0.005"
      SEMANTIC_SCALE_INIT="0.005"
      ;;

    P32|P48|P64)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      local width="${requested#P}"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="phase3_prompt_width"
      HYPOTHESIS="Prompt width sweep; keep only if validation gain is stable."
      PROMPT_CHANNELS="$width"
      ;;

    L_conv|L_ssm|L_avgpool)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      local layout="${requested#L_}"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="phase3_layout_stream"
      HYPOTHESIS="Layout stream sweep; do not claim SSM if conv/avgpool is equal or better."
      LAYOUT_MODE="$layout"
      ;;

    T_lr02_dp10|T_lr03_dp10|T_lr05_dp10)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      local tag="${requested#T_}"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="phase4_training_recipe"
      HYPOTHESIS="Training recipe sweep for final clean model."
      DROP_PATH_RATE="0.10"
      case "$tag" in
        lr02_dp10) ENCODER_LR_MULT="0.2" ;;
        lr03_dp10) ENCODER_LR_MULT="0.3" ;;
        lr05_dp10) ENCODER_LR_MULT="0.5" ;;
      esac
      ;;

    Q_confidence_check)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="diagnostic_only"
      HYPOTHESIS="Diagnostic: confidence routing must earn its complexity before entering the paper."
      USE_CONFIDENCE_ROUTING="1"
      ;;
    Q_gacr_check)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="diagnostic_only"
      HYPOTHESIS="Diagnostic: GACR must earn its complexity before entering the paper."
      CONSISTENCY_ROUTING_MODE="gacr"
      ;;
    Q_ssm_safeoff)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="diagnostic_only"
      HYPOTHESIS="Diagnostic: SSM + GSA/deep branch check; not part of clean mainline by default."
      LAYOUT_MODE="ssm"
      SAFE_MODE="0"
      ;;
    Q_no_prompt_autocorr)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="diagnostic_only"
      HYPOTHESIS="Diagnostic: full model with reliability-aware autocorrelation prompt mixing disabled."
      USE_PROMPT_AUTOCORR="0"
      ;;
    Q_no_directional_edge)
      local requested="$EXPERIMENT"
      define_experiment E5_full_lite
      EXPERIMENT="$requested"
      SAVE_DIR="$UGF_ROOT/$EXPERIMENT"
      PHASE="diagnostic_only"
      HYPOTHESIS="Diagnostic: full model with DEGConv-inspired directional edge refinement disabled."
      USE_DIRECTIONAL_EDGE_REFINE="0"
      ;;
    *)
      echo "[ugf] Unknown experiment: $EXPERIMENT" >&2
      echo "[ugf] Use: bash ugf_lite.sh list" >&2
      exit 1
      ;;
  esac
}

experiment_names() {
  cat <<'EOF'
E0_rgb_only
E1_hha_prompt
E2_rgpr
E3_local_only
E4_semantic_only
E5_full_lite
R_003_003
R_005_003
R_003_005
R_005_005
P32
P48
P64
L_conv
L_ssm
L_avgpool
T_lr02_dp10
T_lr03_dp10
T_lr05_dp10
Q_confidence_check
Q_gacr_check
Q_ssm_safeoff
Q_no_prompt_autocorr
Q_no_directional_edge
EOF
}

phase1_names() {
  cat <<'EOF'
E0_rgb_only
E1_hha_prompt
E2_rgpr
E3_local_only
E4_semantic_only
E5_full_lite
EOF
}

residual_names() {
  cat <<'EOF'
R_003_003
R_005_003
R_003_005
R_005_005
EOF
}

prompt_names() {
  cat <<'EOF'
P32
P48
P64
EOF
}

layout_names() {
  cat <<'EOF'
L_conv
L_ssm
L_avgpool
EOF
}

recipe_names() {
  cat <<'EOF'
T_lr02_dp10
T_lr03_dp10
T_lr05_dp10
EOF
}

diagnostic_names() {
  cat <<'EOF'
Q_confidence_check
Q_gacr_check
Q_ssm_safeoff
Q_no_prompt_autocorr
Q_no_directional_edge
EOF
}

print_experiments() {
  printf "%-20s %-22s %-9s %-8s %-8s %-7s %-5s %-5s %-10s %-8s %-10s %s\n" \
    "experiment" "phase" "layout" "branch" "rec" "conf" "gsa" "rcfr" "autocorr" "dir_edge" "local/semantic" "hypothesis"
  while IFS= read -r name; do
    define_experiment "$name"
    local gsa="off"
    if [[ "$SAFE_MODE" != "1" ]]; then
      gsa="on"
    fi
    local rcfr="off"
    if [[ "$GEOMETRY_ROUTING_MODE" == "rcfr" ]]; then
      rcfr="on"
    fi
    local rec="off"
    if [[ "$USE_PROMPT_RECOVERY" == "1" ]]; then
      rec="$PROMPT_RECOVERY_MODE"
    fi
    printf "%-20s %-22s %-9s %-8s %-8s %-7s %-5s %-5s %-10s %-8s %-10s %s\n" \
      "$name" "$PHASE" "$LAYOUT_MODE" "$FUSION_BRANCH_MODE" "$rec" "$USE_CONFIDENCE_ROUTING" \
      "$gsa" "$rcfr" "$USE_PROMPT_AUTOCORR" "$USE_DIRECTIONAL_EDGE_REFINE" "$LOCAL_SCALE_INIT/$SEMANTIC_SCALE_INIT" "$HYPOTHESIS"
  done < <(experiment_names)
}

best_checkpoint() {
  find "$SAVE_DIR" -maxdepth 1 -type f -name 'best_miou_*.pth' 2>/dev/null | sort | tail -n 1 || true
}

ensure_summary_header() {
  if [[ ! -f "$SUMMARY_CSV" ]]; then
    printf "%s\n" "$SUMMARY_HEADER" > "$SUMMARY_CSV"
    return
  fi
  local current_header
  current_header="$(head -n 1 "$SUMMARY_CSV")"
  if [[ "$current_header" != "$SUMMARY_HEADER" ]]; then
    local legacy_path="${SUMMARY_CSV%.csv}_legacy_$(date +%Y%m%d_%H%M%S).csv"
    mv "$SUMMARY_CSV" "$legacy_path"
    printf "%s\n" "$SUMMARY_HEADER" > "$SUMMARY_CSV"
    echo "[ugf] existing summary header changed; moved old summary to: $legacy_path"
  fi
}

extract_miou() {
  local log_path="$1"
  python - "$log_path" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", encoding="utf-8", errors="ignore").read()
matches = re.findall(r"Final Validation mIoU:\s*([0-9.]+)", text)
print(matches[-1] if matches else "")
PY
}

append_summary() {
  local ckpt="$1"
  local eval_policy="$2"
  local miou="$3"
  ensure_summary_header
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$EXPERIMENT" "$PHASE" "${DATASET_NAME:-SUNRGBD}" "$DECODER_MODE" "$LAYOUT_MODE" "$FUSION_BRANCH_MODE" \
    "$PROMPT_RECOVERY_MODE:$USE_PROMPT_RECOVERY" "$USE_PROMPT_AUTOCORR" "$USE_DIRECTIONAL_EDGE_REFINE" "$USE_CONFIDENCE_ROUTING" "$CONSISTENCY_ROUTING_MODE" "$SAFE_MODE" \
    "$GEOMETRY_ROUTING_MODE" "$PROMPT_CHANNELS" "$LOCAL_SCALE_INIT" "$SEMANTIC_SCALE_INIT" \
    "$ENCODER_LR_MULT" "$DROP_PATH_RATE" "$ckpt" "$eval_policy" "$miou" >> "$SUMMARY_CSV"
}

train_experiment() {
  define_experiment "$1"
  echo "[ugf] train: $EXPERIMENT"
  echo "[ugf] phase: $PHASE"
  echo "[ugf] hypothesis: $HYPOTHESIS"
  echo "[ugf] save_dir: $SAVE_DIR"
  echo "[ugf] gpus: $GPUS"
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
    USE_PROMPT_AUTOCORR="$USE_PROMPT_AUTOCORR" \
    USE_DIRECTIONAL_EDGE_REFINE="$USE_DIRECTIONAL_EDGE_REFINE" \
    SAFE_MODE="$SAFE_MODE" \
    DDP_FIND_UNUSED_PARAMETERS="$DDP_FIND_UNUSED_PARAMETERS" \
    PROMPT_CHANNELS="$PROMPT_CHANNELS" \
    LAYOUT_STATE_DIM="$LAYOUT_STATE_DIM" \
    DECODER_CHANNELS="$DECODER_CHANNELS" \
    ATTENTION_TOKENS="$ATTENTION_TOKENS" \
    LOCAL_SCALE_INIT="$LOCAL_SCALE_INIT" \
    SEMANTIC_SCALE_INIT="$SEMANTIC_SCALE_INIT" \
    ENCODER_LR_MULT="$ENCODER_LR_MULT" \
    DROP_PATH_RATE="$DROP_PATH_RATE" \
    EPOCHS="$EPOCHS" \
    bash run.sh train
}

validate_experiment() {
  define_experiment "$1"
  local ckpt
  ckpt="$(best_checkpoint)"
  if [[ -z "$ckpt" ]]; then
    echo "[ugf] no best checkpoint found in $SAVE_DIR" >&2
    exit 1
  fi

  local eval_policy="single"
  local tta_args=()
  if [[ "${USE_TTA_VAL:-0}" == "1" ]]; then
    eval_policy="tta_0.75_1.0_1.25_flip"
    tta_args+=(--use-tta)
  fi

  local safe_args=()
  if [[ "$SAFE_MODE" == "1" ]]; then
    safe_args+=(--safe-mode)
  fi
  local prompt_args=()
  if [[ "$USE_PROMPT_RECOVERY" != "1" ]]; then
    prompt_args+=(--no-prompt-recovery)
  fi
  local conf_args=()
  if [[ "$USE_CONFIDENCE_ROUTING" != "1" ]]; then
    conf_args+=(--no-confidence-routing)
  fi
  local autocorr_args=()
  if [[ "$USE_PROMPT_AUTOCORR" != "1" ]]; then
    autocorr_args+=(--no-prompt-autocorr)
  fi
  local directional_args=()
  if [[ "$USE_DIRECTIONAL_EDGE_REFINE" != "1" ]]; then
    directional_args+=(--no-directional-edge-refine)
  fi

  local val_log="$SAVE_DIR/ugf_val_${eval_policy}_$(date +%Y%m%d_%H%M%S).log"
  echo "[ugf] val: $EXPERIMENT"
  echo "[ugf] checkpoint: $ckpt"
  echo "[ugf] eval_policy: $eval_policy"
  echo "[ugf] gpus: $GPUS"

  CUDA_VISIBLE_DEVICES="$GPUS" python val.py \
    --data-root "${DATA_ROOT:-/home/pengfei/HTCnet/DataSets}" \
    --dataset-name "${DATASET_NAME:-SUNRGBD}" \
    --ckpt "$ckpt" \
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
    --prompt-channels "$PROMPT_CHANNELS" \
    --layout-state-dim "$LAYOUT_STATE_DIM" \
    --geometry-routing-mode "$GEOMETRY_ROUTING_MODE" \
    --reliability-strength "$RELIABILITY_STRENGTH" \
    --prompt-recovery-mode "$PROMPT_RECOVERY_MODE" \
    --consistency-routing-mode "$CONSISTENCY_ROUTING_MODE" \
    --fusion-branch-mode "$FUSION_BRANCH_MODE" \
    --decoder-channels "$DECODER_CHANNELS" \
    --decoder-mode "$DECODER_MODE" \
    --attention-tokens "$ATTENTION_TOKENS" \
    --local-scale-init "$LOCAL_SCALE_INIT" \
    --semantic-scale-init "$SEMANTIC_SCALE_INIT" \
    --layout-mode "$LAYOUT_MODE" \
    "${safe_args[@]}" \
    "${prompt_args[@]}" \
    "${conf_args[@]}" \
    "${autocorr_args[@]}" \
    "${directional_args[@]}" \
    "${tta_args[@]}" \
    2>&1 | tee "$val_log"

  local miou
  miou="$(extract_miou "$val_log")"
  append_summary "$ckpt" "$eval_policy" "$miou"
  echo "[ugf] summary updated: $SUMMARY_CSV"
  if [[ -n "$miou" ]]; then
    echo "[ugf] final miou: $miou"
  fi
}

report_summary() {
  if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "[ugf] no summary yet: $SUMMARY_CSV"
    exit 0
  fi
  python - "$SUMMARY_CSV" <<'PY'
import csv
import sys

path = sys.argv[1]
rows = []
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            row["_miou"] = float(row.get("miou") or "nan")
        except ValueError:
            row["_miou"] = float("nan")
        rows.append(row)
rows = [r for r in rows if r["_miou"] == r["_miou"]]
rows.sort(key=lambda r: r["_miou"], reverse=True)
print(f"summary: {path}")
for row in rows[:20]:
    print(
        f"{row['miou']:>7}  {row['experiment']:<18} {row['eval_policy']:<22} "
        f"layout={row['layout']:<7} branch={row['fusion_branch']:<8} "
        f"rec={row['prompt_recovery']:<8} autocorr={row.get('prompt_autocorr', '-'):<3} "
        f"dir={row.get('directional_edge', '-'):<3} "
        f"local={row['local_scale']:<6} sem={row['semantic_scale']:<6} "
        f"prompt={row['prompt_channels']}"
    )
PY
}

run_group_train() {
  local group_fn="$1"
  while IFS= read -r name; do
    train_experiment "$name"
  done < <("$group_fn")
}

run_group_val() {
  local group_fn="$1"
  while IFS= read -r name; do
    validate_experiment "$name"
  done < <("$group_fn")
}

run_group_all() {
  local group_fn="$1"
  while IFS= read -r name; do
    train_experiment "$name"
    validate_experiment "$name"
  done < <("$group_fn")
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
  phase1)
    run_group_train phase1_names
    ;;
  val-phase1)
    run_group_val phase1_names
    ;;
  all-phase1)
    run_group_all phase1_names
    ;;
  residual)
    run_group_train residual_names
    ;;
  val-residual)
    run_group_val residual_names
    ;;
  all-residual)
    run_group_all residual_names
    ;;
  prompt)
    run_group_train prompt_names
    ;;
  val-prompt)
    run_group_val prompt_names
    ;;
  all-prompt)
    run_group_all prompt_names
    ;;
  layout)
    run_group_train layout_names
    ;;
  val-layout)
    run_group_val layout_names
    ;;
  all-layout)
    run_group_all layout_names
    ;;
  recipe)
    run_group_train recipe_names
    ;;
  val-recipe)
    run_group_val recipe_names
    ;;
  all-recipe)
    run_group_all recipe_names
    ;;
  diagnostics)
    run_group_train diagnostic_names
    ;;
  val-diagnostics)
    run_group_val diagnostic_names
    ;;
  all-diagnostics)
    run_group_all diagnostic_names
    ;;
  auto-core)
    echo "[ugf] auto-core: phase1 -> residual -> prompt -> layout -> recipe"
    run_group_all phase1_names
    run_group_all residual_names
    run_group_all prompt_names
    run_group_all layout_names
    run_group_all recipe_names
    report_summary
    ;;
  auto-all)
    echo "[ugf] auto-all: auto-core + diagnostics"
    run_group_all phase1_names
    run_group_all residual_names
    run_group_all prompt_names
    run_group_all layout_names
    run_group_all recipe_names
    run_group_all diagnostic_names
    report_summary
    ;;
  report)
    report_summary
    ;;
  *)
    echo "Usage: bash ugf_lite.sh [list|train|val|all|phase1|val-phase1|all-phase1|all-residual|all-prompt|all-layout|all-recipe|all-diagnostics|auto-core|auto-all|report] [experiment]"
    exit 1
    ;;
esac
