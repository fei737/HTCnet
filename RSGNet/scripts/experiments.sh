#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-list}"
EXPERIMENT="${2:-A5_rsgnet}"

default_var() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    printf -v "$name" '%s' "$value"
  fi
}

default_var PROTOCOL "sunrgbd37"
default_var SEED "3407"
default_var RUN_ID "${PROTOCOL}_seed${SEED}"
default_var RUNS_ROOT "$PROJECT_ROOT/../runs_rsgnet"
default_var HHA_DEGRADE_PROB "0.35"
default_var HHA_DEGRADE_MODES "dropout,noise,shift"
default_var RELIABILITY_CLEAN_WEIGHT "0.05"
default_var RELIABILITY_TARGET_TEMPERATURE "0.08"
default_var LAMBDA_RELIABILITY "0.05"
default_var FACTORIZED_HHA_CHANNEL_ORDER "dha"
default_var FACTORIZED_HHA_DIR_NAME "HHA_PFHR"
default_var PFHR_ROBUST_PROFILE "core"

experiment_names() {
  printf '%s\n' \
    A0_rgb_baseline \
    A1_geometry_prompt \
    A2_shallow_geometry \
    A3_deep_semantics \
    A4_stagewise_fusion \
    A5_rsgnet
}

refined_experiment_names() {
  printf '%s\n' \
    B0_refined_rgb \
    B1_refined_prompt \
    B2_refined_shallow \
    B3_refined_semantics \
    B4_refined_stagewise \
    B5_refined_rsgnet
}

factorized_experiment_names() {
  printf '%s\n' \
    C0_factorized_rgb \
    C1_unified_hha \
    C2_independent_hha \
    C3_factorized_static \
    C4_factorized_swapped \
    C5_factorized_routed \
    C6_pfhr_rsgnet \
    C9_hha_channel_relation
}

topology_experiment_names() {
  printf '%s\n' D0_topology_reliability
}

screening_experiment_names() {
  printf '%s\n' \
    C7_final_only_reliability \
    C8_final_only_soft_reliability
}

channel_experiment_names() {
  printf '%s\n' \
    H0_disparity_only \
    H1_height_only \
    H2_angle_only \
    H3_disparity_height
}

describe_experiments() {
  cat <<'EOF'
A0_rgb_baseline       RGB backbone + lightweight decoder
A1_geometry_prompt    Add compact HHA prompt adapters
A2_shallow_geometry   Add shallow C1/C2 geometry-detail fusion
A3_deep_semantics     Add deepest-stage geometry-guided RGB attention
A4_stagewise_fusion   Combine shallow, middle, and deep stage policies
A5_rsgnet             Add calibrated HHA reliability supervision and routing
B0_refined_rgb        Identity-preserving cross-scale RGB baseline
B1_refined_prompt     Refined baseline plus compact HHA prompt adapters
B2_refined_shallow    Add reliability-conditioned pixel fusion at C1/C2
B3_refined_semantics  Add geometry-conditioned linear attention at C4
B4_refined_stagewise  Combine refined shallow, middle, and deep policies
B5_refined_rsgnet     Add single-application reliability routing
C0_factorized_rgb     C-line RGB control with identical cross-scale alignment
C1_unified_hha        Mix canonical D-H-A channels in one geometry stem
C2_independent_hha    Use three independent channel stems without physical relations
C3_factorized_static  D-H layout and D/H/A boundary prompts with fixed stage priors
C4_factorized_swapped Swap shallow/deep physical route priors as a causal control
C5_factorized_routed  Add content-adaptive layout-boundary stage routing
C6_pfhr_rsgnet        Add dual layout/boundary reliability supervision and routing
C9_hha_channel_relation  Add identity-initialized D/H/A channel and pairwise relation routing
C7_final_only_reliability Keep C5 routing; apply reliability only to the final geometry residual
C8_final_only_soft_reliability C7 with a 0.75 reliability floor on valid geometry
D0_topology_reliability  Topology-aware local/global fusion with final reliability gating
R0_raw_depth_control  Capacity-matched metric-depth control for canonical HHA comparison
H0_disparity_only     Capacity-matched disparity-only intervention
H1_height_only        Capacity-matched height-only intervention
H2_angle_only         Capacity-matched angle-only intervention
H3_disparity_height   Capacity-matched disparity-height intervention
EOF
}

configure_experiment() {
  FUSION_MODE="stagewise"
  ARCHITECTURE_VARIANT="legacy"
  GEOMETRY_ENCODING="factorized_routed"
  GEOMETRY_CHANNELS="dha"
  HHA_CHANNEL_ORDER="dha"
  HHA_DIR_NAME="HHA"
  GEOMETRY_SOURCE="hha"
  MAX_DEPTH="10.0"
  DISABLE_RELIABILITY="1"
  case "$EXPERIMENT" in
    A0_rgb_baseline)
      FUSION_MODE="rgb"
      ;;
    A1_geometry_prompt)
      FUSION_MODE="prompt"
      ;;
    A2_shallow_geometry)
      FUSION_MODE="shallow"
      ;;
    A3_deep_semantics)
      FUSION_MODE="deep"
      ;;
    A4_stagewise_fusion)
      FUSION_MODE="stagewise"
      ;;
    A5_rsgnet)
      FUSION_MODE="stagewise"
      DISABLE_RELIABILITY="0"
      ;;
    B0_refined_rgb)
      FUSION_MODE="rgb"
      ARCHITECTURE_VARIANT="refined"
      ;;
    B1_refined_prompt)
      FUSION_MODE="prompt"
      ARCHITECTURE_VARIANT="refined"
      ;;
    B2_refined_shallow)
      FUSION_MODE="shallow"
      ARCHITECTURE_VARIANT="refined"
      ;;
    B3_refined_semantics)
      FUSION_MODE="deep"
      ARCHITECTURE_VARIANT="refined"
      ;;
    B4_refined_stagewise)
      FUSION_MODE="stagewise"
      ARCHITECTURE_VARIANT="refined"
      ;;
    B5_refined_rsgnet)
      FUSION_MODE="stagewise"
      ARCHITECTURE_VARIANT="refined"
      DISABLE_RELIABILITY="0"
      ;;
    C0_factorized_rgb)
      FUSION_MODE="rgb"
      ARCHITECTURE_VARIANT="factorized"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      ;;
    C1_unified_hha)
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="unified"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      ;;
    C2_independent_hha)
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="independent"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      ;;
    C3_factorized_static)
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="factorized_static"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      ;;
    C4_factorized_swapped)
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="factorized_swapped"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      ;;
    C5_factorized_routed)
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="factorized_routed"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      ;;
    C6_pfhr_rsgnet)
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="factorized_routed"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      DISABLE_RELIABILITY="0"
      ;;
    C9_hha_channel_relation)
      FUSION_MODE="stagewise"
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="factorized_channel_routed"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      DISABLE_RELIABILITY="0"
      # The new candidate is intended for a complete per-epoch curve.
      VAL_INTERVAL="${VAL_INTERVAL:-1}"
      ;;
    C7_final_only_reliability)
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="factorized_final"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      DISABLE_RELIABILITY="0"
      ;;
    C8_final_only_soft_reliability)
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="factorized_final_soft"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      DISABLE_RELIABILITY="0"
      ;;
    D0_topology_reliability)
      FUSION_MODE="topology"
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="factorized_final_soft"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      DISABLE_RELIABILITY="0"
      ;;
    R0_raw_depth_control)
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="unified"
      GEOMETRY_SOURCE="depth"
      MAX_DEPTH="8.0"
      HHA_CHANNEL_ORDER="dha"
      ;;
    H0_disparity_only|H1_height_only|H2_angle_only|H3_disparity_height)
      ARCHITECTURE_VARIANT="factorized"
      GEOMETRY_ENCODING="factorized_static"
      HHA_CHANNEL_ORDER="$FACTORIZED_HHA_CHANNEL_ORDER"
      HHA_DIR_NAME="$FACTORIZED_HHA_DIR_NAME"
      case "$EXPERIMENT" in
        H0_disparity_only) GEOMETRY_CHANNELS="d" ;;
        H1_height_only) GEOMETRY_CHANNELS="h" ;;
        H2_angle_only) GEOMETRY_CHANNELS="a" ;;
        H3_disparity_height) GEOMETRY_CHANNELS="dh" ;;
      esac
      ;;
    *)
      echo "[RSGNet] Unknown experiment: $EXPERIMENT" >&2
      describe_experiments >&2
      exit 1
      ;;
  esac

  SAVE_DIR="$RUNS_ROOT/$EXPERIMENT/$RUN_ID"
  if [[ "$ARCHITECTURE_VARIANT" == "refined" || "$ARCHITECTURE_VARIANT" == "factorized" ]]; then
    default_var LOCAL_SCALE_INIT "0.10"
    default_var SEMANTIC_SCALE_INIT "0.10"
    default_var EARLY_STOPPING_PATIENCE "0"
    default_var VAL_INTERVAL "100000"
  fi
  if [[ "$DISABLE_RELIABILITY" == "1" ]]; then
    HHA_DEGRADE_PROB="0.0"
    LAMBDA_RELIABILITY="0.0"
  fi

  export PROTOCOL SEED SAVE_DIR FUSION_MODE ARCHITECTURE_VARIANT GEOMETRY_ENCODING
  export GEOMETRY_CHANNELS HHA_CHANNEL_ORDER HHA_DIR_NAME GEOMETRY_SOURCE MAX_DEPTH
  export DISABLE_RELIABILITY
  if [[ "$ARCHITECTURE_VARIANT" == "refined" || "$ARCHITECTURE_VARIANT" == "factorized" ]]; then
    export LOCAL_SCALE_INIT SEMANTIC_SCALE_INIT EARLY_STOPPING_PATIENCE VAL_INTERVAL
  fi
  export HHA_DEGRADE_PROB HHA_DEGRADE_MODES
  export RELIABILITY_CLEAN_WEIGHT RELIABILITY_TARGET_TEMPERATURE LAMBDA_RELIABILITY
}

run_once() {
  local action="$1"
  configure_experiment
  echo "[RSGNet] experiment=$EXPERIMENT | run=$RUN_ID | action=$action"
  bash "$PROJECT_ROOT/scripts/run_rsgnet.sh" "$action"
}

run_suite() {
  local action="$1"
  local name
  while IFS= read -r name; do
    bash "$PROJECT_ROOT/scripts/experiments.sh" "$action" "$name"
  done < <(experiment_names)
}

run_refined_suite() {
  local action="$1"
  local name
  while IFS= read -r name; do
    bash "$PROJECT_ROOT/scripts/experiments.sh" "$action" "$name"
  done < <(refined_experiment_names)
}

run_factorized_suite() {
  local action="$1"
  local name
  while IFS= read -r name; do
    bash "$PROJECT_ROOT/scripts/experiments.sh" "$action" "$name"
  done < <(factorized_experiment_names)
}

run_channel_suite() {
  local action="$1"
  local name
  while IFS= read -r name; do
    bash "$PROJECT_ROOT/scripts/experiments.sh" "$action" "$name"
  done < <(channel_experiment_names)
}

run_topology_suite() {
  local action="$1"
  local name
  while IFS= read -r name; do
    bash "$PROJECT_ROOT/scripts/experiments.sh" "$action" "$name"
  done < <(topology_experiment_names)
}

run_robustness() {
  EXPERIMENT="$1"
  configure_experiment
  local mode severity tag
  while read -r mode severity tag; do
    EVAL_HHA_DEGRADE_MODE="$mode" \
    EVAL_HHA_DEGRADE_SEVERITY="$severity" \
    METRICS_JSON="$SAVE_DIR/robust_$tag.json" \
      bash "$PROJECT_ROOT/scripts/run_rsgnet.sh" eval
  done <<'EOF'
clean 0.0 clean
dropout 0.1 dropout_010
dropout 0.3 dropout_030
dropout 0.5 dropout_050
noise 8 noise_008
noise 16 noise_016
noise 24 noise_024
shift 4 shift_004
shift 8 shift_008
shift 12 shift_012
EOF
}

run_physical_robustness_model() (
  local experiment="$1"
  local clean_only="${2:-0}"
  local clean_hha_dir="$FACTORIZED_HHA_DIR_NAME"
  local mode severity tag hha_dir

  EXPERIMENT="$experiment"
  FACTORIZED_HHA_DIR_NAME="$clean_hha_dir"
  configure_experiment
  echo "[RSGNet] physical robustness | experiment=$EXPERIMENT | input=clean"
  EVAL_HHA_DEGRADE_MODE=clean \
  EVAL_HHA_DEGRADE_SEVERITY=0.0 \
  METRICS_JSON="$SAVE_DIR/physical_clean.json" \
    bash "$PROJECT_ROOT/scripts/run_rsgnet.sh" eval

  if [[ "$clean_only" == "1" ]]; then
    return
  fi

  while read -r mode severity tag hha_dir; do
    FACTORIZED_HHA_DIR_NAME="$hha_dir"
    configure_experiment
    echo "[RSGNet] physical robustness | experiment=$EXPERIMENT | tag=$tag | raw_mode=$mode | raw_severity=$severity"
    EVAL_HHA_DEGRADE_MODE=clean \
    EVAL_HHA_DEGRADE_SEVERITY=0.0 \
    METRICS_JSON="$SAVE_DIR/physical_${tag}.json" \
      bash "$PROJECT_ROOT/scripts/run_rsgnet.sh" eval
  done < <(
    PFHR_ROBUST_PROFILE="$PFHR_ROBUST_PROFILE" \
      bash "$PROJECT_ROOT/scripts/prepare_pfhr_robustness.sh" list-machine
  )
)

run_physical_robustness_suite() {
  run_physical_robustness_model "C0_factorized_rgb" 1
  run_physical_robustness_model "C5_factorized_routed"
  run_physical_robustness_model "C6_pfhr_rsgnet"
}

case "$MODE" in
  list)
    describe_experiments
    ;;
  config)
    configure_experiment
    echo "experiment=$EXPERIMENT"
    echo "variant=$ARCHITECTURE_VARIANT geometry=$GEOMETRY_ENCODING channels=$GEOMETRY_CHANNELS fusion=$FUSION_MODE reliability=$((1 - DISABLE_RELIABILITY))"
    echo "hha_channel_order=$HHA_CHANNEL_ORDER"
    echo "geometry_source=$GEOMETRY_SOURCE hha_dir=$HHA_DIR_NAME"
    echo "run_id=$RUN_ID save_dir=$SAVE_DIR"
    echo "local_scale=${LOCAL_SCALE_INIT:-default} semantic_scale=${SEMANTIC_SCALE_INIT:-default}"
    echo "val_interval=${VAL_INTERVAL:-default} early_stopping_patience=${EARLY_STOPPING_PATIENCE:-default}"
    ;;
  train|resume|finetune|eval|routes)
    run_once "$MODE"
    ;;
  all)
    run_once train
    run_once eval
    ;;
  suite)
    run_suite all
    ;;
  train-suite)
    run_suite train
    ;;
  eval-suite)
    run_suite eval
    ;;
  refined-suite)
    run_refined_suite all
    ;;
  train-refined-suite)
    run_refined_suite train
    ;;
  eval-refined-suite)
    run_refined_suite eval
    ;;
  factorized-suite)
    run_factorized_suite all
    ;;
  train-factorized-suite)
    run_factorized_suite train
    ;;
  eval-factorized-suite)
    run_factorized_suite eval
    ;;
  topology-suite)
    run_topology_suite all
    ;;
  train-topology-suite)
    run_topology_suite train
    ;;
  eval-topology-suite)
    run_topology_suite eval
    ;;
  channel-suite)
    run_channel_suite all
    ;;
  train-channel-suite)
    run_channel_suite train
    ;;
  eval-channel-suite)
    run_channel_suite eval
    ;;
  robust)
    run_robustness "A5_rsgnet"
    ;;
  robust-refined)
    run_robustness "B5_refined_rsgnet"
    ;;
  robust-factorized)
    run_robustness "C6_pfhr_rsgnet"
    ;;
  prepare-pfhr-robustness)
    PFHR_ROBUST_PROFILE="$PFHR_ROBUST_PROFILE" \
      bash "$PROJECT_ROOT/scripts/prepare_pfhr_robustness.sh" generate
    ;;
  verify-pfhr-robustness)
    PFHR_ROBUST_PROFILE="$PFHR_ROBUST_PROFILE" \
      bash "$PROJECT_ROOT/scripts/prepare_pfhr_robustness.sh" verify
    ;;
  analyze-pfhr-robustness)
    PFHR_ROBUST_PROFILE="$PFHR_ROBUST_PROFILE" \
      bash "$PROJECT_ROOT/scripts/prepare_pfhr_robustness.sh" analyze
    ;;
  robust-physical-factorized)
    run_physical_robustness_suite
    ;;
  *)
    echo "Usage: bash scripts/experiments.sh [list|config|train|resume|finetune|eval|all|suite|train-suite|eval-suite|refined-suite|train-refined-suite|eval-refined-suite|factorized-suite|train-factorized-suite|eval-factorized-suite|topology-suite|train-topology-suite|eval-topology-suite|channel-suite|train-channel-suite|eval-channel-suite|robust|robust-refined|robust-factorized|prepare-pfhr-robustness|verify-pfhr-robustness|analyze-pfhr-robustness|robust-physical-factorized] [experiment]" >&2
    exit 1
    ;;
esac
