#!/usr/bin/env bash
# Fixed-work A/B with energy: each argument is a gpu.cu variant (a path, or "rev:<git rev>").
# Each is built as a 512-iteration bench binary (1 pipeline, small biomes, golden), then the
# variants run alternately ROUNDS times. Prints the main stage times, the total, and the GPU
# energy for the run (nvidia-smi's cumulative counter). The GPU is always at a power cap
# (285 W, or 100 W in agile-power's eco mode), so joules per fixed work is the measure that
# tracks real throughput; times alone move with clocks.
# usage (on the node): tools/ab.sh rev:HEAD src/gpu.cu
set -uo pipefail
cd "$(dirname "$0")/.."
ROUNDS=${ROUNDS:-2}; ITERS=${ITERS:-512}; R=/tmp/ab
rm -rf $R; mkdir -p $R
i=0; names=()
for v in "$@"; do
  i=$((i + 1)); n="v$i"; names+=($n)
  rsync -a --exclude .git --exclude search --exclude 'main-*' ./ $R/$n/
  if [[ $v == rev:* ]]; then git show "${v#rev:}:src/gpu.cu" > $R/$n/src/gpu.cu; else cp "$v" $R/$n/src/gpu.cu; fi
  (cd $R/$n && flock /tmp/commission-build.lock make -B -s NO_NET=1 MAX_ITERS=$ITERS PRINT_INTERVAL=$ITERS ${EXTRA:-} >/dev/null 2>&1; cp main-sb $R/$n.bin)
  echo "$n = $v"
done
energy() { python3 tools/gpu_energy.py; }
for r in $(seq $ROUNDS); do for n in "${names[@]}"; do
  e0=$(energy)
  $R/$n.bin --pipelines 1 --start 1000000000000 --output /dev/null > $R/$n.log 2>&1
  e1=$(energy)
  awk -v n=$n -v r=$r -v j=$(( (e1 - e0) / 1000 )) '
    /^(filter_seeds|filter_gradvecs_1|filter_2_01a|filter_gradvecs_2) /{t[$1]=$3}
    /^total /{tot=$3; out=$10}
    END{printf "%s r%s  seeds %7.1f  gv1 %7.1f  2_01a %7.1f  gv2 %6.1f  total %7.1f ms  %5d J  (final count %s)\n",
        n, r, t["filter_seeds"], t["filter_gradvecs_1"], t["filter_2_01a"], t["filter_gradvecs_2"], tot, j, out}' $R/$n.log
done; done
echo "mode: $(agile status 2>/dev/null | grep -o '"mode": "[a-z]*"' | cut -d'"' -f4)"
