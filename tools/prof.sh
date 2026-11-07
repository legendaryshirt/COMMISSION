#!/usr/bin/env bash
# Profile one kernel (first launch after skipping SKIP) with Nsight Compute.
# usage (on the node): tools/prof.sh <kernel-regex> [metrics...]   (default: a work/stall set)
set -euo pipefail
cd "$(dirname "$0")/.."
K=$1; shift || true
M=${*:-"smsp__thread_inst_executed.sum,smsp__inst_executed.sum,gpu__time_duration.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,regex:smsp__average_warps_issue_stalled_(short_scoreboard|wait|not_selected|math_pipe_throttle|long_scoreboard|barrier|mio_throttle|no_instruction|branch_resolving)_per_issue_active.ratio"}
T=$(mktemp -d)
{ flock 9; make -B -s NO_NET=1 MAX_ITERS=1 PRINT_INTERVAL=1 ${EXTRA:-} >/dev/null; cp main-sb $T/b; } 9>/tmp/commission-build.lock
cd $T
sudo /usr/local/cuda/bin/ncu --kernel-name-base demangled -k "regex:$K" -c 1 --metrics "$M" ./b --pipelines 1 --start 1000000000000 --output /dev/null 2>/dev/null | sed -n '/Metric Name/,$p'
