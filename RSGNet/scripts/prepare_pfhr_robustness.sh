#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-list}"

default_var() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    printf -v "$name" '%s' "$value"
  fi
}

default_var PYTHON_BIN "/home/pengfei/miniconda3/envs/PFseg/bin/python"
default_var PREPROCESS_SCRIPT "/home/pengfei/HTCnet/getDATA/prepare_sunrgbd.py/SUN2HHA.py"
default_var RAW_ROOT "/home/pengfei/datasets_Original/SUNRGBD/SUNRGBD_Raw"
default_var OUT_ROOT "/home/pengfei/HTCnet/DataSets/SUNRGBD"
default_var WORKERS "8"
default_var SUBSET "test"
default_var CORRUPTION_SEED "3407"
default_var PFHR_ROBUST_PROFILE "core"
default_var MAX_IMAGES "0"
default_var PFHR_LOG_DIR "$OUT_ROOT/pfhr_robustness_logs"
default_var PFHR_ANALYSIS_SAMPLES "101"

physical_grid() {
  case "$PFHR_ROBUST_PROFILE" in
    core)
      printf '%s\n' \
        "dropout 0.30 dropout_p30 HHA_PFHR_dropout_p30" \
        "noise 0.05 noise_m050 HHA_PFHR_noise_m050" \
        "shift 8 shift_px08 HHA_PFHR_shift_px08" \
        "tilt 5 tilt_deg05 HHA_PFHR_tilt_deg05"
      ;;
    full)
      printf '%s\n' \
        "dropout 0.10 dropout_p10 HHA_PFHR_dropout_p10" \
        "dropout 0.30 dropout_p30 HHA_PFHR_dropout_p30" \
        "dropout 0.50 dropout_p50 HHA_PFHR_dropout_p50" \
        "noise 0.02 noise_m020 HHA_PFHR_noise_m020" \
        "noise 0.05 noise_m050 HHA_PFHR_noise_m050" \
        "noise 0.10 noise_m100 HHA_PFHR_noise_m100" \
        "shift 4 shift_px04 HHA_PFHR_shift_px04" \
        "shift 8 shift_px08 HHA_PFHR_shift_px08" \
        "shift 12 shift_px12 HHA_PFHR_shift_px12" \
        "tilt 2 tilt_deg02 HHA_PFHR_tilt_deg02" \
        "tilt 5 tilt_deg05 HHA_PFHR_tilt_deg05" \
        "tilt 10 tilt_deg10 HHA_PFHR_tilt_deg10"
      ;;
    *)
      echo "[PFHR] Unknown PFHR_ROBUST_PROFILE: $PFHR_ROBUST_PROFILE" >&2
      exit 1
      ;;
  esac
}

expected_count() {
  local count
  case "$SUBSET" in
    all) count=10335 ;;
    train) count=5285 ;;
    test) count=5050 ;;
    *)
      echo "[PFHR] SUBSET must be all, train, or test; got $SUBSET" >&2
      exit 1
      ;;
  esac
  if [[ "$MAX_IMAGES" -gt 0 && "$MAX_IMAGES" -lt "$count" ]]; then
    count="$MAX_IMAGES"
  fi
  echo "$count"
}

generate_grid() {
  local mode severity tag directory image_count expected log_file
  expected="$(expected_count)"
  mkdir -p "$PFHR_LOG_DIR"
  while read -r mode severity tag directory; do
    if [[ -d "$OUT_ROOT/$directory" && -f "$OUT_ROOT/$directory/manifest.json" ]]; then
      image_count="$(find "$OUT_ROOT/$directory" -maxdepth 1 -type f -name '*.png' -print | wc -l)"
      if [[ "$image_count" -eq "$expected" ]]; then
        echo "[PFHR] reuse verified $directory ($image_count PNGs)"
        continue
      fi
    fi
    echo "[PFHR] generate tag=$tag mode=$mode severity=$severity subset=$SUBSET"
    log_file="$PFHR_LOG_DIR/${tag}.log"
    if ! "$PYTHON_BIN" "$PREPROCESS_SCRIPT" \
        --raw-root "$RAW_ROOT" \
        --out-root "$OUT_ROOT" \
        --hha-dir-name "$directory" \
        --depth-dir-name Depth \
        --workers "$WORKERS" \
        --subset "$SUBSET" \
        --max-images "$MAX_IMAGES" \
        --skip-depth \
        --corruption-mode "$mode" \
        --corruption-severity "$severity" \
        --corruption-seed "$CORRUPTION_SEED" >"$log_file" 2>&1; then
      echo "[PFHR] generation failed; tail of $log_file:" >&2
      tr '\r' '\n' <"$log_file" | tail -n 20 >&2
      exit 1
    fi
    tr '\r' '\n' <"$log_file" | tail -n 3

    image_count="$(find "$OUT_ROOT/$directory" -maxdepth 1 -type f -name '*.png' -print | wc -l)"
    if [[ "$image_count" -ne "$expected" ]]; then
      echo "[PFHR] $directory contains $image_count PNGs; expected $expected" >&2
      exit 1
    fi
    if [[ ! -f "$OUT_ROOT/$directory/manifest.json" ]]; then
      echo "[PFHR] Missing manifest: $OUT_ROOT/$directory/manifest.json" >&2
      exit 1
    fi
    echo "[PFHR] verified $directory ($image_count PNGs)"
  done < <(physical_grid)
}

verify_grid() {
  local mode severity tag directory image_count expected
  expected="$(expected_count)"
  while read -r mode severity tag directory; do
    if [[ ! -d "$OUT_ROOT/$directory" ]]; then
      echo "[PFHR] missing directory for tag=$tag: $OUT_ROOT/$directory" >&2
      exit 1
    fi
    image_count="$(find "$OUT_ROOT/$directory" -maxdepth 1 -type f -name '*.png' -print | wc -l)"
    if [[ "$image_count" -ne "$expected" || ! -f "$OUT_ROOT/$directory/manifest.json" ]]; then
      echo "[PFHR] incomplete tag=$tag dir=$directory png=$image_count expected=$expected" >&2
      exit 1
    fi
    echo "[PFHR] ok tag=$tag dir=$directory png=$image_count"
  done < <(physical_grid)
}

analyze_grid() {
  local mode severity tag directory
  local analysis_args=()
  while read -r mode severity tag directory; do
    analysis_args+=(--degraded "$tag=$directory")
  done < <(physical_grid)
  "$PYTHON_BIN" "$PROJECT_ROOT/tools/analyze_pfhr_corruptions.py" \
    --data-root "$OUT_ROOT" \
    --clean-dir HHA_PFHR \
    --max-samples "$PFHR_ANALYSIS_SAMPLES" \
    --output "$OUT_ROOT/pfhr_robustness_${PFHR_ROBUST_PROFILE}.json" \
    "${analysis_args[@]}"
}

case "$ACTION" in
  list)
    echo "mode severity tag directory"
    physical_grid
    ;;
  list-machine)
    physical_grid
    ;;
  generate)
    generate_grid
    ;;
  verify)
    verify_grid
    ;;
  analyze)
    analyze_grid
    ;;
  smoke)
    OUT_ROOT="${SMOKE_OUT_ROOT:-/tmp/rsgnet_pfhr_physical_smoke}"
    PFHR_LOG_DIR="$OUT_ROOT/pfhr_robustness_logs"
    MAX_IMAGES=1
    WORKERS=1
    generate_grid
    ;;
  *)
    echo "Usage: bash scripts/prepare_pfhr_robustness.sh [list|generate|verify|analyze|smoke]" >&2
    exit 1
    ;;
esac
