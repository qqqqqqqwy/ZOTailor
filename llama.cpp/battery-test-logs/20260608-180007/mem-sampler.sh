#!/system/bin/sh
umask 000

OUT=/data/local/tmp/mem-samples.csv
FLAG=/data/local/tmp/mem-sampler.run
TARGET="$1"
SAMPLE_US=500000

sample_sleep() {
  if command -v usleep >/dev/null 2>&1; then
    usleep "$1"
  else
    sleep 0.5
  fi
}

is_wrapper_comm() {
  case "$1" in
    sh|su|toybox|pgrep|grep|awk|sed|tr|cat|head|sleep)
      return 0
      ;;
  esac
  return 1
}

echo "timestamp_ns,pid,comm,VmRSS_kB,VmHWM_kB,cmdline" > "$OUT"
while [ -f "$FLAG" ]; do
  best_pid=
  best_comm=
  best_cmd=
  best_rss=-1
  best_hwm=0

  for proc_dir in /proc/[0-9]*; do
    [ -r "$proc_dir/status" ] || continue

    pid=${proc_dir#/proc/}
    comm=$(cat "$proc_dir/comm" 2>/dev/null | tr -d '[:space:]')
    is_wrapper_comm "$comm" && continue

    cmd=$(tr '\000,' '  ' < "$proc_dir/cmdline" 2>/dev/null | tr '\r\n' '  ')
    match=0
    [ "$comm" = "$TARGET" ] && match=1
    case "$cmd " in
      *"/$TARGET "*|*"./$TARGET "*|*" $TARGET "*|*"bin/$TARGET "*)
        match=1
        ;;
    esac
    [ "$match" = 1 ] || continue

    vals=$(awk '
      /^VmRSS:/ { rss = $2 }
      /^VmHWM:/ { hwm = $2 }
      END { printf "%d %d", rss + 0, hwm + 0 }
    ' "$proc_dir/status" 2>/dev/null)
    set -- $vals
    rss=${1:-0}
    hwm=${2:-0}

    if [ "$rss" -gt "$best_rss" ]; then
      best_pid=$pid
      best_comm=$comm
      best_cmd=$cmd
      best_rss=$rss
      best_hwm=$hwm
    fi
  done

  if [ -n "$best_pid" ]; then
    ts=$(date +%s%N 2>/dev/null || date +%s000000000)
    printf '%s,%s,%s,%s,%s,%s\n' \
      "$ts" "$best_pid" "$best_comm" "$best_rss" "$best_hwm" "$best_cmd" >> "$OUT"
  fi

  sample_sleep "$SAMPLE_US"
done
