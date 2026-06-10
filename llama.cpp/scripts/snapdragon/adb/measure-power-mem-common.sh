#!/usr/bin/env bash

# Shared helpers for Android battery/current and process-memory sampling.

write_sampler_scripts() {
  local out_dir="${OUT_DIR:?OUT_DIR is required}"
  local dev_tmp="${DEV_TMP:-/data/local/tmp}"

  cat > "$out_dir/power-sampler.sh" <<'POWER_EOF'
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
POWER_EOF

  cat > "$out_dir/mem-sampler.sh" <<'MEM_EOF'
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
MEM_EOF

  echo "Pushing sampler scripts to device..."
  adb_push "$out_dir/power-sampler.sh" "$dev_tmp/power-sampler.sh" >/dev/null
  adb_push "$out_dir/mem-sampler.sh"   "$dev_tmp/mem-sampler.sh"   >/dev/null
}

start_samplers() {
  local mem_target="${1:?process name is required}"
  local dev_tmp="${DEV_TMP:-/data/local/tmp}"

  adb_shell "rm -f $dev_tmp/power-samples.csv $dev_tmp/mem-samples.csv" >/dev/null 2>&1 || true
  adb_shell "touch $dev_tmp/sampler.run $dev_tmp/mem-sampler.run" >/dev/null

  adb_shell "su -c 'sh $dev_tmp/power-sampler.sh'" >/dev/null 2>&1 &
  POWER_SAMPLER_HOST_PID=$!
  adb_shell "su -c 'sh $dev_tmp/mem-sampler.sh $mem_target'" >/dev/null 2>&1 &
  MEM_SAMPLER_HOST_PID=$!
}

stop_samplers() {
  local power_csv="${1:?power csv path is required}"
  local mem_csv="${2:?memory csv path is required}"
  local dev_tmp="${DEV_TMP:-/data/local/tmp}"

  adb_shell "rm -f $dev_tmp/sampler.run $dev_tmp/mem-sampler.run" >/dev/null 2>&1 || true
  wait "${POWER_SAMPLER_HOST_PID:-}" 2>/dev/null || true
  wait "${MEM_SAMPLER_HOST_PID:-}" 2>/dev/null || true

  adb_pull "$dev_tmp/power-samples.csv" "$power_csv" >/dev/null 2>&1 || true
  adb_pull "$dev_tmp/mem-samples.csv"   "$mem_csv"   >/dev/null 2>&1 || true
}

calc_power_metrics() {
  local power_csv="$1"
  local current_unit="${CURRENT_UNIT:-auto}"

  if [[ ! -s "$power_csv" ]]; then
    printf '0.000 0.0 0.0 0.0 0 0.0 unknown\n'
    return
  fi

  awk -F, -v requested_unit="$current_unit" '
    NR == 1 {
      current_col = 3
      next
    }
    {
      ts = $1 + 0
      v = $2 + 0
      i = $current_col + 0
      abs_i = i < 0 ? -i : i
      rows[++n] = ts "," v "," abs_i
      sum_i += abs_i
      if (abs_i > max_i) max_i = abs_i
    }
    END {
      if (n < 2) {
        printf "0.000 0.0 0.0 0.0 %d 0.0 unknown\n", n + 0
        exit
      }

      unit = requested_unit
      if (unit == "" || unit == "auto") {
        avg_i = sum_i / n
        unit = (avg_i > 0 && avg_i < 10000 && max_i < 10000) ? "mA" : "uA"
      }
      scale = unit == "mA" ? 1e-9 : 1e-12

      for (idx = 1; idx <= n; idx++) {
        split(rows[idx], f, ",")
        ts = f[1] + 0
        v = f[2] + 0
        i = f[3] + 0
        p = v * i * scale

        if (idx == 1) {
          first_ts = ts
          prev_ts = ts
          prev_p = p
          peak_p = p
          last_ts = ts
          continue
        }

        dt = (ts - prev_ts) / 1e9
        if (dt < 0 || dt > 60) {
          prev_ts = ts
          prev_p = p
          last_ts = ts
          continue
        }
        energy += (p + prev_p) / 2 * dt
        sum_dt += dt
        used_dt++
        if (p > peak_p) peak_p = p
        prev_ts = ts
        prev_p = p
        last_ts = ts
      }

      total = (last_ts - first_ts) / 1e9
      avg_p = total > 0 ? energy / total : 0
      e_mWh = energy / 3.6
      mean_dt_ms = used_dt > 0 ? sum_dt / used_dt * 1000 : 0
      printf "%.3f %.1f %.1f %.1f %d %.1f %s\n", e_mWh, avg_p * 1000, peak_p * 1000, total, n, mean_dt_ms, unit
    }
  ' "$power_csv"
}

calc_mem_metrics() {
  local mem_csv="$1"

  if [[ ! -s "$mem_csv" ]]; then
    printf '0 0 0 NA NA\n'
    return
  fi

  awk -F, '
    NR == 1 {
      has_pid = ($2 == "pid")
      next
    }
    {
      if (has_pid) {
        pid = $2
        comm = $3
        rss = $4 + 0
        hwm = $5 + 0
      } else {
        pid = "NA"
        comm = "legacy"
        rss = $2 + 0
        hwm = $3 + 0
      }

      if (rss > p_rss) p_rss = rss
      if (hwm > p_hwm) {
        p_hwm = hwm
        peak_pid = pid
        peak_comm = comm
      }
      n++
    }
    END {
      if (peak_pid == "") peak_pid = "NA"
      if (peak_comm == "") peak_comm = "NA"
      printf "%d %d %d %s %s\n", p_hwm + 0, p_rss + 0, n + 0, peak_pid, peak_comm
    }
  ' "$mem_csv"
}

kb_to_mb() {
  awk -v x="${1:-0}" 'BEGIN { printf "%.1f", x / 1024 }'
}
