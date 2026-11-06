#!/usr/bin/env bash
# Resumable long-running search that never revisits a seed range.
# Ledger: search/ranges.tsv (gitignored, so cnode sync never overwrites it on the node):
#   closed_utc  build_key  first_logical  end_logical(exclusive)  islands_found
# Ranges are in *logical* seed index (what --start takes). Under SEED_ORDER=golden, index n is
# seed (n*GOLDEN)^SILVER, a bijection over all 2^64 seeds. Ranges only make sense within one
# build key (order + biomes + bound), so each key resumes from its own furthest end.
# Benchmarks use logical 1e12 upward; searches default to 2^62 upward to stay clear of them.
# usage: tools/search.sh [extra main args, e.g. --size 11000000]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p search
SEED_ORDER=${SEED_ORDER:-golden}; LARGE_BIOMES=${LARGE_BIOMES:-0}; UNBOUND=${UNBOUND:-0}
KEY="${SEED_ORDER}-$([ "$LARGE_BIOMES" = 1 ] && echo lb || echo sb)-$([ "$UNBOUND" = 1 ] && echo unbound || echo bounded)"
LEDGER=search/ranges.tsv; LOG=search/current.log; BATCH=$((1 << 28)); DEFAULT_START=$((1 << 62))

close_session() {   # record the range covered by a finished or killed previous session
  [ -f "$LOG" ] || return 0
  local key first end found
  key=$(cat search/current.key); first=$(awk '/^Starting from/{print $3; exit}' "$LOG")
  local st; st=$(awk '/^Start seed: /{gsub(",","",$3); print $3, $7}' "$LOG" | tail -1)
  end=""; [ -n "$st" ] && end=$(( ${st% *} + ${st#* } ))   # bash int64: awk doubles lose bits near 2^62
  if [ -z "$end" ]; then   # killed hard: last completed batch printed in the stats table
    end=$(awk '/^start_seed = /{s=$3} END{if (s!="") print s}' "$LOG")
    [ -n "$end" ] && end=$((end + BATCH))
  fi
  found=$(grep -c . search/current.found 2>/dev/null || true); found=${found:-0}
  [ -n "$first" ] && [ -n "$end" ] && printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$key" "$first" "$end" "$found" >> "$LEDGER"
  cat search/current.found >> search/found.txt 2>/dev/null || true
  mv "$LOG" "search/session-$(date +%s).log"; rm -f search/current.found
}
close_session
[ "${1:-}" = close ] && exit 0

next=$(awk -F'\t' -v k="$KEY" '$2==k && $4>m {m=$4} END{if (m!="") print m}' "$LEDGER" 2>/dev/null || true)
next=${next:-$DEFAULT_START}
echo "$KEY" > search/current.key
bin=main-$([ "$UNBOUND" = 1 ] && echo u)$([ "$LARGE_BIOMES" = 1 ] && echo lb || echo sb)
# Build under the lock that tools/bench.sh also takes, and run a private copy, so a benchmark
# build in the same tree can never replace the binary the search runs.
{
  flock 9
  make -s NO_NET=1 SEED_ORDER="$SEED_ORDER" LARGE_BIOMES="$LARGE_BIOMES" UNBOUND="$UNBOUND" PRINT_INTERVAL=64 MAX_ITERS=0 -B >/dev/null
  cp "$bin" search/search-bin
} 9>/tmp/commission-build.lock
echo "search $KEY from logical $next"
exec ./search/search-bin --start "$next" --threads 4 --output search/current.found "$@" > "$LOG" 2>&1
