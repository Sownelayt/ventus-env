#!/bin/bash

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

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
export MAX_AREA=${MAX_AREA:-100000}
export CRITICAL_RANGE_NS=${CRITICAL_RANGE_NS:-0.20}
export MAX_CORES=${MAX_CORES:-12}
export DC_TIMEOUT_SECONDS=${DC_TIMEOUT_SECONDS:-0}
export VARIANT=${VARIANT:-v3_7_line40_payload6_zerowire_1500}
export DESIGN_TOP=${DESIGN_TOP:-tma}
export TIMING_MODEL=${TIMING_MODEL:-zero_wire}

if [ "$MAX_CORES" -gt 12 ]; then
  echo "MAX_CORES=$MAX_CORES exceeds the per-project limit of 12" >&2
  exit 2
fi
if ! [[ "$DC_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "DC_TIMEOUT_SECONDS must be a non-negative integer" >&2
  exit 2
fi

LOG=${DC_LOG:-dc_${VARIANT}_n12_tt1v85c_1500.log}
printf 'RUNNING\n' > dc_exit.status
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
printf '%s\n' "$rc" > dc_exit.status
exit "$rc"
