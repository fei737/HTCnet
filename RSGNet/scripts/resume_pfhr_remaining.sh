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

default_var GPUS "0,1,2,3"
default_var RUN_ID "pfhr_seed3407"
default_var SEED "3407"
default_var PROTOCOL "sunrgbd37"
default_var RUNS_ROOT "$PROJECT_ROOT/../runs_rsgnet"
default_var NUM_WORKERS "4"
default_var BATCH_SIZE "4"
default_var TTA_SCALES "0.5,0.75,1.0,1.25,1.5"
default_var PYTHON_BIN "/home/pengfei/miniconda3/envs/PFseg/bin/python"
default_var DRY_RUN "0"
default_var RUN_CHANNEL_SUITE "0"

export GPUS RUN_ID SEED PROTOCOL RUNS_ROOT NUM_WORKERS BATCH_SIZE TTA_SCALES

is_complete_json() {
  local path="$1"
  [[ -s "$path" ]] && "$PYTHON_BIN" -c \
    'import json, sys; json.load(open(sys.argv[1], "r", encoding="utf-8"))' "$path" \
    >/dev/null 2>&1
}

run_physical_eval() {
  local experiment="$1"
  local tag="$2"
  local hha_dir="$3"
  local metrics="$RUNS_ROOT/$experiment/$RUN_ID/physical_${tag}.json"

  if is_complete_json "$metrics"; then
    echo "[PFHR resume] skip completed: $experiment/$tag"
    return
  fi

  echo "[PFHR resume] evaluate: $experiment/$tag | HHA=$hha_dir | GPUs=$GPUS"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  FACTORIZED_HHA_DIR_NAME="$hha_dir" \
  FACTORIZED_HHA_CHANNEL_ORDER="dha" \
  EVAL_HHA_DEGRADE_MODE="clean" \
  EVAL_HHA_DEGRADE_SEVERITY="0.0" \
  METRICS_JSON="$metrics" \
    bash scripts/experiments.sh eval "$experiment"
}

run_route_analysis() {
  local experiment="$1"
  local output="$RUNS_ROOT/$experiment/$RUN_ID/route_diagnostics.json"

  if is_complete_json "$output"; then
    echo "[PFHR resume] skip completed route analysis: $experiment"
    return
  fi

  echo "[PFHR resume] route analysis: $experiment"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  ROUTE_JSON="$output" \
    bash scripts/experiments.sh routes "$experiment"
}

run_channel_experiment() {
  local experiment="$1"
  local save_dir="$RUNS_ROOT/$experiment/$RUN_ID"
  local metrics="$save_dir/metrics.json"
  local checkpoint=""

  if [[ -d "$save_dir" ]]; then
    checkpoint="$(find "$save_dir" -maxdepth 1 -type f -name 'best_miou_*.pth' -print 2>/dev/null | sort | tail -n 1)"
  fi
  if [[ -z "$checkpoint" ]]; then
    echo "[PFHR resume] train channel ablation: $experiment"
    if [[ "$DRY_RUN" != "1" ]]; then
      bash scripts/experiments.sh train "$experiment"
    fi
  else
    echo "[PFHR resume] skip trained channel ablation: $experiment"
  fi

  if is_complete_json "$metrics"; then
    echo "[PFHR resume] skip evaluated channel ablation: $experiment"
    return
  fi
  echo "[PFHR resume] evaluate channel ablation: $experiment"
  if [[ "$DRY_RUN" != "1" ]]; then
    bash scripts/experiments.sh eval "$experiment"
  fi
}

# C5 entries already completed are checked and skipped. This makes the script
# safe to rerun after another interruption.
run_physical_eval "C5_factorized_routed" "clean" "HHA_PFHR"
run_physical_eval "C5_factorized_routed" "dropout_p30" "HHA_PFHR_dropout_p30"
run_physical_eval "C5_factorized_routed" "noise_m050" "HHA_PFHR_noise_m050"
run_physical_eval "C5_factorized_routed" "shift_px08" "HHA_PFHR_shift_px08"
run_physical_eval "C5_factorized_routed" "tilt_deg05" "HHA_PFHR_tilt_deg05"

run_physical_eval "C6_pfhr_rsgnet" "clean" "HHA_PFHR"
run_physical_eval "C6_pfhr_rsgnet" "dropout_p30" "HHA_PFHR_dropout_p30"
run_physical_eval "C6_pfhr_rsgnet" "noise_m050" "HHA_PFHR_noise_m050"
run_physical_eval "C6_pfhr_rsgnet" "shift_px08" "HHA_PFHR_shift_px08"
run_physical_eval "C6_pfhr_rsgnet" "tilt_deg05" "HHA_PFHR_tilt_deg05"

run_route_analysis "C5_factorized_routed"
run_route_analysis "C6_pfhr_rsgnet"

if [[ "$RUN_CHANNEL_SUITE" == "1" ]]; then
  run_channel_experiment "H0_disparity_only"
  run_channel_experiment "H1_height_only"
  run_channel_experiment "H2_angle_only"
  run_channel_experiment "H3_disparity_height"
fi

echo "[PFHR resume] requested remaining PFHR work completed."
