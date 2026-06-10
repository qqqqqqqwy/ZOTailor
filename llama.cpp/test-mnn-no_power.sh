#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-battery-test-logs/$RUN_ID}"
SUMMARY="${SUMMARY:-./mnn.txt}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-600}"
STEPS="${STEPS:-300}"

DEV_WORKDIR="/data/local/tmp/zotailor/mnn"
TARGET_BIN="zo_lora_fa_sst2"

mkdir -p "$OUT_DIR"

ADB=(adb)
if [[ -n "${S:-}" ]]; then ADB+=(-s "$S"); fi
if [[ -n "${H:-}" ]]; then ADB+=(-H "$H"); fi

adb_shell() { "${ADB[@]}" shell "$@"; }

echo "Output directory: $OUT_DIR"
echo "Summary file:     $SUMMARY"
echo "Checking adb device..."
"${ADB[@]}" get-state >/dev/null

run_one() {
  local label="$1"
  local config="$2"

  local run_log="$OUT_DIR/run-$label.log"

  echo
  echo "===== [$label] starting ====="

  local start_epoch end_epoch duration
  start_epoch=$(date +%s)

  set +e
  adb_shell "cd $DEV_WORKDIR && ./$TARGET_BIN \
    --config $config \
    --train-data sst2/train.tsv \
    --valid-data sst2/dev.tsv \
    --steps $STEPS \
    --eval-step -1 \
    --threads -1 \
    --precision high \
    --memory normal \
    --attention-mode 8 \
    --force-full-causal-mask true \
    --force-float-weight false \
    --b-param-precision fp32 \
    --verbose" 2>&1 | tee "$run_log"
  local run_status=${PIPESTATUS[0]}
  set -e

  end_epoch=$(date +%s)
  duration=$(( end_epoch - start_epoch ))

  local now_iso
  now_iso=$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)

  {
    echo "============="
    echo "[$now_iso] $label  (exit=$run_status, duration=${duration}s)"
    echo "Config:            $config"
    echo "Steps:             $STEPS"
    echo "--- last 50 lines of $(basename "$run_log") ---"
    tail -n 50 "$run_log" || true
  } >> "$SUMMARY"

  echo "===== [$label] done (exit=$run_status) ====="
}

RUNS=(
  "tiny:tiny_int4/config_lora_fa.json"
  "llama:llama_int4/config_lora_fa.json"
  "gemma:gemma_int4/config_lora_fa.json"
)

for idx in "${!RUNS[@]}"; do
  if (( idx > 0 )); then
    echo "Sleeping ${SLEEP_BETWEEN}s before next run..."
    sleep "$SLEEP_BETWEEN"
  fi
  entry="${RUNS[$idx]}"
  run_one "${entry%%:*}" "${entry#*:}"
done

echo
echo "All runs complete."
echo "Summary appended to: $SUMMARY"
echo "Per-run artifacts in: $OUT_DIR"
