#!/system/bin/sh
umask 000
OUT=/data/local/tmp/power-samples.csv
FLAG=/data/local/tmp/sampler.run
VNODE=/sys/class/power_supply/battery/voltage_now
INODE=/sys/class/power_supply/battery/current_now
echo "timestamp_ns,voltage_uV,current_uA" > "$OUT"
while [ -f "$FLAG" ]; do
  ts=$(date +%s%N)
  v=$(cat "$VNODE" 2>/dev/null)
  i=$(cat "$INODE" 2>/dev/null)
  echo "$ts,$v,$i" >> "$OUT"
  sleep 0.1
done
