#!/bin/bash

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

if [ -r variant.env ]; then
  source variant.env
fi
if [ -r /eda/synopsys/env.sh ]; then
  source /eda/synopsys/env.sh
fi
if ! command -v dc_shell >/dev/null 2>&1 ||
    [[ $(command -v dc_shell) != /eda/synopsys/tools_patched/* ]]; then
  source /eda/synopsys/env_crack.sh
fi
DC_SHELL=/eda/synopsys/tools_patched/syn/X-2025.06/bin/dc_shell
if [ ! -x "$DC_SHELL" ]; then
  echo "patched dc_shell is unavailable: $DC_SHELL" >&2
  exit 2
fi
set -u

export SYNTH_FREQ_MHZ=${SYNTH_FREQ_MHZ:-1500}
export REPORT_FREQ_MHZ=${REPORT_FREQ_MHZ:-1500}
export MAX_CORES=${MAX_CORES:-12}
export DC_TIMEOUT_SECONDS=${DC_TIMEOUT_SECONDS:-0}
export DC_LICENSE_RETRIES=${DC_LICENSE_RETRIES:-1}
export DC_LICENSE_RETRY_INTERVAL=${DC_LICENSE_RETRY_INTERVAL:-60}
export VARIANT=${VARIANT:-v3_4_release_40x6_desc4}
export DESIGN_TOP=${DESIGN_TOP:-tma}

if [ "$MAX_CORES" -gt 12 ]; then
  echo "MAX_CORES=$MAX_CORES exceeds the project limit of 12" >&2
  exit 2
fi
if ! [[ "$DC_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "DC_TIMEOUT_SECONDS must be a non-negative integer" >&2
  exit 2
fi

LOG=${DC_LOG:-dc_${VARIANT}_n12_tt1v85c_1500.log}
RETRY_LOG=${DC_RETRY_LOG:-dc_${VARIANT}_license_retry.log}
attempt=0
printf 'RUNNING\n' > dc_exit.status
while true; do
  attempt=$((attempt + 1))
  printf '%s attempt=%d\n' "$(date '+%F %T')" "$attempt" >> "$RETRY_LOG"
  set +e
  if [ "$DC_TIMEOUT_SECONDS" -eq 0 ]; then
    "$DC_SHELL" -64bit -f dc/run_dc_n12_tt1v85c_1500.tcl \
      > "$LOG" 2>&1
  else
    timeout --signal=TERM --kill-after=60 "$DC_TIMEOUT_SECONDS" \
      "$DC_SHELL" -64bit -f dc/run_dc_n12_tt1v85c_1500.tcl \
      > "$LOG" 2>&1
  fi
  rc=$?
  set -e
  if [ "$rc" -eq 0 ] && grep -Eq '^(Error:|Fatal:)' "$LOG"; then
    rc=2
  fi
  if [ "$rc" -eq 0 ]; then
    break
  fi
  if ! grep -q 'DCSH-1' "$LOG" ||
      [ "$attempt" -ge "$DC_LICENSE_RETRIES" ]; then
    break
  fi
  sleep "$DC_LICENSE_RETRY_INTERVAL"
done
printf '%s\n' "$rc" > dc_exit.status
exit "$rc"
