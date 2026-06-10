#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SLEEP_BETWEEN="${SLEEP_BETWEEN:-600}"

SCRIPTS=(
  # ./test-ours-no_power.sh
  ./test-1-no_power.sh
  # ./test-1+2-no_power.sh
  ./test-1+3-no_power.sh
  # ./test-llama.cpp-no_power.sh
  ./test-executorch-no_power.sh
  ./test-mnn-no_power.sh
)

for idx in "${!SCRIPTS[@]}"; do
  if (( idx > 0 )); then
    echo "[$(date -Iseconds 2>/dev/null || date)] sleeping ${SLEEP_BETWEEN}s before next script..."
    sleep "$SLEEP_BETWEEN"
  fi
  script="${SCRIPTS[$idx]}"
  echo "[$(date -Iseconds 2>/dev/null || date)] ===== launching $script ====="
  set +e
  "$script"
  status=$?
  set -e
  echo "[$(date -Iseconds 2>/dev/null || date)] ===== $script exited with $status ====="
done

echo
echo "All scripts complete."
