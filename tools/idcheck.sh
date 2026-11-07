#!/usr/bin/env bash
# Set-identity check: a reference git revision (default HEAD) vs the working tree. Both get
# tools/dump_patch.py, run 32 iterations from the same start for small biomes, large biomes and
# unbound, and the sorted candidate sets of 5 mid-funnel stages are compared by SHA-256.
# Run on the compute node: cnode run 'tools/idcheck.sh [ref]'. Pause the search first for speed.
set -euo pipefail
cd "$(dirname "$0")/.."
REF=${1:-HEAD}; R=/tmp/idc
rm -rf $R; mkdir -p $R
git worktree remove --force $R/ref 2>/dev/null || true
git worktree prune
git worktree add -q --detach $R/ref "$REF"
rsync -a --exclude .git --exclude search --exclude 'main-*' ./ $R/new/
for d in ref new; do python3 tools/dump_patch.py $R/$d/src/gpu.cu; done
fail=0
for v in "0 0" "1 0" "0 1"; do
  set -- $v
  for d in ref new; do
    ( cd $R/$d && flock /tmp/commission-build.lock make -B -s NO_NET=1 LARGE_BIOMES=$1 UNBOUND=$2 PRINT_INTERVAL=32 >/dev/null 2>&1
      b=$(ls -t main-* | head -1)
      COMMISSION_DUMP=$R/$d-$1$2.txt COMMISSION_DUMP_ITERS=32 ./$b --pipelines 1 --start 1000000000000 --threads 1 --output /dev/null >/dev/null 2>&1 || true )
  done
  for st in 2_01a gv2 2_01b 2a 2b; do
    a=$(grep "^$st " $R/ref-$1$2.txt | sort | sha256sum | cut -c1-12); n=$(grep -c "^$st " $R/ref-$1$2.txt || true)
    b=$(grep "^$st " $R/new-$1$2.txt | sort | sha256sum | cut -c1-12); m=$(grep -c "^$st " $R/new-$1$2.txt || true)
    ok=DIFF; [ "$a" = "$b" ] && [ "$n" = "$m" ] && [ "$n" -gt 0 ] && ok=SAME
    [ $ok = SAME ] || fail=1
    echo "LB=$1 UB=$2 $st ref=$n/$a new=$m/$b $ok"
  done
done
git worktree remove --force $R/ref
[ $fail = 0 ] && echo "ALL 15 SETS IDENTICAL" || { echo "MISMATCH"; exit 1; }
