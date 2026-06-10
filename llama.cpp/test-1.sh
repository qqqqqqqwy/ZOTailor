#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-battery-test-logs/$RUN_ID}"
SUMMARY="${SUMMARY:-./1.txt}"
FAKE_UNPLUG="${FAKE_UNPLUG:-1}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-400}"
STEPS="${STEPS:-50}"

DEV_TMP="/data/local/tmp"
TARGET_BIN="llama-zoo-sst2-zo-lora"

mkdir -p "$OUT_DIR"

ADB=(adb)
if [[ -n "${S:-}" ]]; then ADB+=(-s "$S"); fi
if [[ -n "${H:-}" ]]; then ADB+=(-H "$H"); fi

adb_shell() { "${ADB[@]}" shell "$@"; }
adb_push()  { "${ADB[@]}" push "$@"; }
adb_pull()  { "${ADB[@]}" pull "$@"; }

source "$ROOT_DIR/scripts/snapdragon/adb/measure-power-mem-common.sh"

cleanup() {
  adb_shell "rm -f $DEV_TMP/sampler.run $DEV_TMP/mem-sampler.run" >/dev/null 2>&1 || true
  if [[ "$FAKE_UNPLUG" == "1" ]]; then
    adb_shell "dumpsys battery reset" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Output directory: $OUT_DIR"
echo "Summary file:     $SUMMARY"
echo "Checking adb device..."
"${ADB[@]}" get-state >/dev/null

write_sampler_scripts

if [[ "$FAKE_UNPLUG" == "1" ]]; then
  echo "Forcing Android battery service to unplug mode..."
  adb_shell "dumpsys battery unplug" >/dev/null
fi

run_one() {
  local label="$1"
  local lora="$2"
  local model="$3"

  local run_log="$OUT_DIR/run-$label.log"
  local power_csv="$OUT_DIR/power-$label.csv"
  local mem_csv="$OUT_DIR/mem-$label.csv"

  echo
  echo "===== [$label] starting ====="

  start_samplers "$TARGET_BIN"

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

  stop_samplers "$power_csv" "$mem_csv"

  local energy_mWh avg_p_mW peak_p_mW total_s p_cnt mean_dt_ms current_unit
  read -r energy_mWh avg_p_mW peak_p_mW total_s p_cnt mean_dt_ms current_unit <<<"$(calc_power_metrics "$power_csv")"

  local peak_hwm_kb peak_rss_kb m_cnt peak_pid peak_comm
  read -r peak_hwm_kb peak_rss_kb m_cnt peak_pid peak_comm <<<"$(calc_mem_metrics "$mem_csv")"

  local peak_hwm_mb peak_rss_mb
  peak_hwm_mb=$(kb_to_mb "$peak_hwm_kb")
  peak_rss_mb=$(kb_to_mb "$peak_rss_kb")

  local now_iso
  now_iso=$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)

  {
    echo "============="
    echo "[$now_iso] $label  (exit=$run_status, duration=${duration}s)"
    echo "Backend:           zoo-htp coop (PROMPT_FA=1)"
    echo "Model:             $model"
    echo "LoRA:              $lora"
    echo "Steps:             $STEPS"
    echo "Peak memory VmHWM: ${peak_hwm_mb} MB"
    echo "Peak memory VmRSS: ${peak_rss_mb} MB  (sampled, ${m_cnt} samples)"
    echo "Sampled process:   pid=${peak_pid}, comm=${peak_comm}"
    echo "Energy:            ${energy_mWh} mWh"
    echo "Avg power:         ${avg_p_mW} mW"
    echo "Peak power:        ${peak_p_mW} mW"
    echo "Current unit:      ${current_unit} (auto; override with CURRENT_UNIT=uA or mA)"
    echo "Power samples:     ${p_cnt} (target 100 ms, mean dt = ${mean_dt_ms} ms, total ${total_s} s)"
    echo "--- last 50 lines of $(basename "$run_log") ---"
    tail -n 50 "$run_log" || true
  } >> "$SUMMARY"

  echo "===== [$label] done (exit=$run_status, VmHWM=${peak_hwm_mb} MB, energy=${energy_mWh} mWh) ====="
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
