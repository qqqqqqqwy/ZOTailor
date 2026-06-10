#!/system/bin/sh
umask 000
OUT=/data/local/tmp/mem-samples.csv
FLAG=/data/local/tmp/mem-sampler.run
TARGET=lora_fa_runner_sst2
echo "timestamp_ns,VmRSS_kB,VmHWM_kB" > "$OUT"
while [ -f "$FLAG" ]; do
  pid=$(pgrep -f "$TARGET" 2>/dev/null | head -n 1)
  if [ -z "$pid" ]; then
    pid=$(pidof lora_fa_runner_sst2 2>/dev/null | awk '{print $1}')
  fi
  if [ -n "$pid" ] && [ -r "/proc/$pid/status" ]; then
    ts=$(date +%s%N)
    rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    hwm=$(awk '/^VmHWM:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    if [ -n "$rss" ] && [ -n "$hwm" ]; then
      echo "$ts,$rss,$hwm" >> "$OUT"
    fi
  fi
  sleep 0.5
done
