#!/usr/bin/env bash
# record_temps.sh — log CPU/GPU temperatures to CSV
# Usage:
#   chmod +x record_temps.sh
#   ./record_temps.sh [-i SECONDS] [-o OUTPUT.csv]
#
# Examples:
#   ./record_temps.sh
#   ./record_temps.sh -i 2 -o temps.csv

set -euo pipefail

INTERVAL=5
OUTFILE="temps.csv"

while getopts ":i:o:" opt; do
  case "$opt" in
    i) INTERVAL="${OPTARG}" ;;
    o) OUTFILE="${OPTARG}" ;;
    *) echo "Usage: $0 [-i SECONDS] [-o OUTPUT.csv]" >&2; exit 1 ;;
  esac
done

log() { printf "%s\n" "$*" >&2; }

write_header_if_needed() {
  if [[ ! -f "$OUTFILE" ]]; then
    echo "timestamp,cpu_c,gpu_c" > "$OUTFILE"
  fi
}

get_cpu_temp() {
  # 1) Try lm-sensors
  if command -v sensors >/dev/null 2>&1; then
    local t
    t="$(sensors 2>/dev/null | awk '
      /Package id 0:|Tctl:|Tdie:|CPU Temperature:|CPU:/ {
        if (match($0, /\+?([0-9]+(\.[0-9]+)?)°C/, a)) { print a[1]; exit }
      }
      /^Core [0-9]+:/ {
        if (match($0, /\+?([0-9]+(\.[0-9]+)?)°C/, a)) { temps[++n]=a[1] }
      }
      END {
        if (n>0) {
          max=temps[1]; for(i=2;i<=n;i++) if (temps[i]>max) max=temps[i];
          print max
        }
      }'
    )"
    if [[ -n "${t:-}" ]]; then echo "$t"; return 0; fi
  fi

  # 2) Try common sysfs thermal zones
  for tz in /sys/class/thermal/thermal_zone*; do
    [[ -e "$tz/type" && -e "$tz/temp" ]] || continue
    local type; type="$(tr -d '\0' <"$tz/type" 2>/dev/null || true)"
    if [[ "$type" =~ [Cc][Pp][Uu]|x86_pkg_temp|acpitz|soc|pch ]]; then
      local mdeg; mdeg="$(cat "$tz/temp" 2>/dev/null || true)"
      if [[ -n "$mdeg" ]]; then awk -v m="$mdeg" 'BEGIN{printf "%.1f", m/1000}'; return 0; fi
    fi
  done

  # 3) Fallback: first hwmon temp
  for f in /sys/class/hwmon/hwmon*/temp*_input; do
    [[ -e "$f" ]] || continue
    local mdeg; mdeg="$(cat "$f" 2>/dev/null || true)"
    if [[ -n "$mdeg" ]]; then awk -v m="$mdeg" 'BEGIN{printf "%.1f", m/1000}'; return 0; fi
  done

  echo "NA"
}

get_gpu_temp() {
  # NVIDIA
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
      | paste -sd'/' - | sed 's/^$/NA/'
    return 0
  fi
  # AMD ROCm (rocm-smi / amd-smi)
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showtemp 2>/dev/null \
      | awk '/GPU\[/{ if (match($0, /([0-9]+)C/, a)) t[++i]=a[1] } END{ for(j=1;j<=i;j++) printf j==1?t[j]:"/"t[j] }'
    [[ ${PIPESTATUS[1]} -eq 0 ]] || true
    return 0
  fi
  if command -v amd-smi >/dev/null 2>&1; then
    amd-smi -t 2>/dev/null \
      | awk 'match($0, /([0-9]+)C/, a){ t[++i]=a[1] } END{ for(j=1;j<=i;j++) printf j==1?t[j]:"/"t[j] }'
    return 0
  fi
  # Generic via lm-sensors (amdgpu/nouveau/radeon)
  if command -v sensors >/dev/null 2>&1; then
    sensors 2>/dev/null | awk '
      /amdgpu-|nouveau|radeon|GPU/ && /°C/ {
        if (match($0, /\+?([0-9]+(\.[0-9]+)?)°C/, a)) { print a[1]; exit }
      }'
    return 0
  fi

  echo "NA"
}

trap 'log "Stopping..."; exit 0' INT TERM

write_header_if_needed
log "Logging every ${INTERVAL}s to ${OUTFILE} (Ctrl-C to stop)."

while :; do
  ts="$(date -Is)"
  cpu="$(get_cpu_temp)"
  gpu="$(get_gpu_temp)"
  echo "${ts},${cpu},${gpu}" >> "$OUTFILE"
  sleep "$INTERVAL"
done
