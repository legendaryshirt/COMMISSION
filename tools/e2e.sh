#!/usr/bin/env bash
# Real-world A/B: each argument is a gpu.cu variant (path relative to the repo, or a git rev
# prefixed "rev:"). Each is built as the search binary (golden, small biomes), then run for
# SECS seconds with 2 pipelines, alternating ROUNDS times. Prints seeds checked per run and the
# GPU mode/clock/power, because the power cap (and agile-power's eco mode) dominate the numbers.
# usage (on the node): tools/e2e.sh rev:HEAD src/gpu.cu
set -uo pipefail
cd "$(dirname "$0")/.."
SECS=${SECS:-60}; ROUNDS=${ROUNDS:-2}; R=/tmp/e2e
rm -rf $R; mkdir -p $R
i=0; names=()
for v in "$@"; do
  i=$((i + 1)); n="v$i"; names+=($n)
  rsync -a --exclude .git --exclude search --exclude 'main-*' ./ $R/$n/
  if [[ $v == rev:* ]]; then git show "${v#rev:}:src/gpu.cu" > $R/$n/src/gpu.cu; else cp "$v" $R/$n/src/gpu.cu; fi
  (cd $R/$n && flock /tmp/commission-build.lock make -B -s NO_NET=1 PRINT_INTERVAL=64 >/dev/null 2>&1; cp main-sb $R/$n.bin)
  echo "$n = $v"
done
for r in $(seq $ROUNDS); do for n in "${names[@]}"; do
  timeout -s INT $SECS $R/$n.bin --pipelines 2 --threads 4 --start 2000000000000 --output /dev/null > $R/$n.log 2>&1
  seeds=$(grep -a 'Total seeds checked' $R/$n.log | tail -1 | sed 's/.*checked: //')
  echo "$n r$r seeds=$seeds  $(awk -v s=$seeds -v t=$SECS 'BEGIN{printf "%.2f G/s", s/t/1e9}')  mode=$(agile status 2>/dev/null | grep -o '"mode": "[a-z]*"' | cut -d'"' -f4) $(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader)"
done; done
