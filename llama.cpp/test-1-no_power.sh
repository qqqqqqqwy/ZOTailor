#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-battery-test-logs/$RUN_ID}"
SUMMARY="${SUMMARY:-./1.txt}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-300}"
STEPS="${STEPS:-300}"

TARGET_BIN="llama-zoo-sst2-zo-lora"

mkdir -p "$OUT_DIR"

echo "Output directory: $OUT_DIR"
echo "Summary file:     $SUMMARY"

run_one() {
  local label="$1"
  local lora="$2"
  local model="$3"

  local run_log="$OUT_DIR/run-$label.log"

  echo
  echo "===== [$label] starting ====="

  local start_epoch end_epoch duration
  start_epoch=$(date +%s)

  set +e
  B=zoo-htp PROMPT_FA=1 ./scripts/snapdragon/adb/run-tool.sh "$TARGET_BIN" \
    --mode coop --pipeline false --antithetic false \
    --lora-exec fused-htp --device HTP0 \
    --lora "$lora" \
    -m "$model" \
    -ngl 99 -fa on --eval-step -1 --steps "$STEPS" 2>&1 | tee "$run_log"
  local run_status=${PIPESTATUS[0]}
  set -e

  end_epoch=$(date +%s)
  duration=$(( end_epoch - start_epoch ))

  local now_iso
  now_iso=$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)

  {
    echo "============="
    echo "[$now_iso] $label  (exit=$run_status, duration=${duration}s)"
    echo "Backend:           zoo-htp coop (PROMPT_FA=1, pipeline=false, antithetic=false)"
    echo "Model:             $model"
    echo "LoRA:              $lora"
    echo "Steps:             $STEPS"
    echo "--- last 50 lines of $(basename "$run_log") ---"
    tail -n 50 "$run_log" || true
  } >> "$SUMMARY"

  echo "===== [$label] done (exit=$run_status) ====="
}

RUNS=(
  "tiny:/data/local/tmp/gguf/Tiny-lora.gguf:/data/local/tmp/gguf/TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf"
  "llama:/data/local/tmp/gguf/Llama-lora.gguf:/data/local/tmp/gguf/Llama-3.2-3B-Q4_0.gguf"
  "gemma:/data/local/tmp/gguf/gemma-lora.gguf:/data/local/tmp/gguf/gemma-3-4b-it-Q4_0.gguf"
)

for idx in "${!RUNS[@]}"; do
  if (( idx > 0 )); then
    echo "Sleeping ${SLEEP_BETWEEN}s before next run..."
    sleep "$SLEEP_BETWEEN"
  fi
  entry="${RUNS[$idx]}"
  label="${entry%%:*}"
  rest="${entry#*:}"
  lora="${rest%%:*}"
  model="${rest#*:}"
  run_one "$label" "$lora" "$model"
done

echo
echo "All runs complete."
echo "Summary appended to: $SUMMARY"
echo "Per-run artifacts in: $OUT_DIR"
