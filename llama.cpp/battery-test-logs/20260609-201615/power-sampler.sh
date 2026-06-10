#!/system/bin/sh
umask 000

OUT=/data/local/tmp/power-samples.csv
FLAG=/data/local/tmp/sampler.run
VNODE=/sys/class/power_supply/battery/voltage_now
INODE=/sys/class/power_supply/battery/current_now
SAMPLE_US=100000

sample_sleep() {
  if command -v usleep >/dev/null 2>&1; then
    usleep "$1"
  else
    sleep 0.1
  fi
}

echo "timestamp_ns,voltage_uV,current_raw" > "$OUT"
while [ -f "$FLAG" ]; do
  ts=$(date +%s%N 2>/dev/null || date +%s000000000)
  v=$(cat "$VNODE" 2>/dev/null | tr -d '[:space:]')
  i=$(cat "$INODE" 2>/dev/null | tr -d '[:space:]')

  case "$v" in ''|*[!0-9-]*) v=0 ;; esac
  case "$i" in ''|*[!0-9-]*) i=0 ;; esac

  printf '%s,%s,%s\n' "$ts" "$v" "$i" >> "$OUT"
  sample_sleep "$SAMPLE_US"
done
