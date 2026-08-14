#!/bin/bash

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

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

export DESIGN_TOP=${DESIGN_TOP:-tma}
export SYNTH_FREQ_MHZ=${SYNTH_FREQ_MHZ:-1500}
export REPORT_FREQ_MHZ=${REPORT_FREQ_MHZ:-1500}
export MAX_CORES=${MAX_CORES:-12}
export DC_TIMEOUT_SECONDS=${DC_TIMEOUT_SECONDS:-7200}
export DC_LICENSE_RETRIES=${DC_LICENSE_RETRIES:-1}
export DC_LICENSE_RETRY_INTERVAL=${DC_LICENSE_RETRY_INTERVAL:-60}
export VARIANT=${VARIANT:-$(basename "$ROOT")}

if [ "$MAX_CORES" -gt 12 ]; then
  echo "MAX_CORES=$MAX_CORES exceeds the project limit of 12" >&2
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
  timeout --signal=TERM --kill-after=60 "$DC_TIMEOUT_SECONDS" \
    "$DC_SHELL" -64bit -f dc/run_dc_n12_tt1v85c_1500.tcl > "$LOG" 2>&1
  rc=$?
  set -e

  if [ "$rc" -eq 0 ] && grep -Eq '^(Error:|Fatal:)' "$LOG"; then
    printf '%s DC emitted an error despite exit status 0\n' \
      "$(date '+%F %T')" >> "$RETRY_LOG"
    rc=2
  fi
  if [ "$rc" -eq 0 ]; then
    break
  fi
  if ! grep -q 'DCSH-1' "$LOG" ||
      [ "$attempt" -ge "$DC_LICENSE_RETRIES" ]; then
    break
  fi
  printf '%s license unavailable; retrying in %ss\n' \
    "$(date '+%F %T')" "$DC_LICENSE_RETRY_INTERVAL" >> "$RETRY_LOG"
  sleep "$DC_LICENSE_RETRY_INTERVAL"
done

printf '%s\n' "$rc" > dc_exit.status
exit "$rc"
