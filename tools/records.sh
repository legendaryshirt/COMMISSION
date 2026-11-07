#!/usr/bin/env bash
# Record test: does a build still find every known record island?
# Each record's seed is searched directly (linear-order build, --start <seed>, one 2^28-seed batch,
# full CPU verification), and the output must contain that seed with x and z within TOL blocks of
# the sheet. Acceptance rule (Louis, 2026-11-07) for any change that alters what gets filtered:
# reject if it misses even 1 of the top 25, or 2 or more of the whole list.
# usage (on the node): tools/records.sh <variant> [records.tsv]
#   variant: a gpu.cu path, "rev:<git rev>", or "." for the working tree as is.
set -uo pipefail
cd "$(dirname "$0")/.."
V=${1:-.}; LIST=${2:-tools/records/sb.tsv}; TOL=${TOL:-2000}
LB=0; UB=0
case "$(basename "$LIST" .tsv)" in lb) LB=1;; usb) UB=1;; ulb) LB=1; UB=1;; esac
R=/tmp/records; rm -rf $R; mkdir -p $R
rsync -a --exclude .git --exclude search --exclude 'main-*' ./ $R/src/
if [[ $V == rev:* ]]; then git show "${V#rev:}:src/gpu.cu" > $R/src/src/gpu.cu; elif [ "$V" != . ]; then cp "$V" $R/src/src/gpu.cu; fi
(cd $R/src && flock /tmp/commission-build.lock make -B -s NO_NET=1 SEED_ORDER=linear LARGE_BIOMES=$LB UNBOUND=$UB MAX_ITERS=1 PRINT_INTERVAL=1 ${EXTRA:-} >/dev/null 2>&1)
BIN=$(ls -t $R/src/main-* | head -1)
miss_top=0; miss_all=0; n=0
while IFS=$'\t' read -r rank seed x z size; do
  [[ $rank == \#* ]] && continue
  n=$((n + 1))
  out=$R/out-$rank.txt
  "$BIN" --pipelines 1 --threads 8 --start "$seed" --output "$out" >/dev/null 2>&1
  # Several candidates of one island can be printed with different measured sizes: report the
  # largest within TOL of the sheet's position.
  hit=$(awk -v s="$seed" -v x="$x" -v z="$z" -v t=$TOL '
    $1 == s { dx = (x == "-") ? 0 : $2 - x; dz = $3 - z; if (dx < 0) dx = -dx; if (dz < 0) dz = -dz;
              if (dx <= t && dz <= t && $4 > best) { best = $4; line = sprintf("%s %s %s (dx %d dz %d)", $2, $3, $4, dx, dz) } }
    END { if (line != "") print line }' "$out" 2>/dev/null)
  if [ -n "$hit" ]; then
    echo "#$rank $seed found: $hit  (sheet size $size)"
  else
    echo "#$rank $seed MISSED  (sheet $x $z $size; output: $(grep -c . "$out" 2>/dev/null || echo 0) lines)"
    miss_all=$((miss_all + 1)); [ "$rank" -le 25 ] && miss_top=$((miss_top + 1))
  fi
done < "$LIST"
verdict=PASS; { [ $miss_top -ge 1 ] || [ $miss_all -ge 2 ]; } && verdict=REJECT
echo "records: $n, missed $miss_all (top 25: $miss_top) -> $verdict"
