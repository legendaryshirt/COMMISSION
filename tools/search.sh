#!/usr/bin/env bash
# Resumable long-running search that never revisits a seed range.
# Everything worth keeping goes to one small append-only log, search/search.log (gitignored, so
# cnode sync never overwrites it on the node), written by tools/search_log.py:
#   <utc> range  <type> <order> <first> <end>   logical seeds [first, end) fully searched
#   <utc> island <type> <seed> <x> <z> <area>
# type: sb / lb / sub / lub (small/large biomes, bounded/unbound). Ranges are in *logical*
# seed index (what --start takes). Under SEED_ORDER=golden, index n is seed (n*GOLDEN)^SILVER,
# a bijection over all 2^64 seeds, so ranges only make sense within one type + order: each
# resumes from its own furthest end. A session writes a range checkpoint every 5 minutes and
# its exact end on a graceful stop, so even a hard kill re-checks at most a few minutes and
# never leaves a gap. The binary's per-second stage tables are not kept: search/status.json
# holds the latest one (overwritten each minute), and search/session.txt the startup banner
# and any errors of the current session.
# Benchmarks use logical 1e12 upward; searches default to 2^62 upward to stay clear of them.
# usage: tools/search.sh [extra main args, e.g. --size 11000000]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p search
SEED_ORDER=${SEED_ORDER:-golden}; LARGE_BIOMES=${LARGE_BIOMES:-0}; UNBOUND=${UNBOUND:-0}
TYPE=$([ "$UNBOUND" = 1 ] && echo u)$([ "$LARGE_BIOMES" = 1 ] && echo lb || echo sb)
TYPE=${TYPE/usb/sub}; TYPE=${TYPE/ulb/lub}
P=${PIPELINES:-2}; BATCH=$((1 << 28)); DEFAULT_START=$((1 << 62)); LOG=search/search.log
[ "${1:-}" = close ] && exit 0   # kept for old callers: sessions now close themselves

# Resume from the furthest end for this type + order. Compared in Python: exact 64-bit ints.
next=$(python3 - "$LOG" "$TYPE" "$SEED_ORDER" "$DEFAULT_START" <<'EOF'
import sys
path, typ, order, default = sys.argv[1:5]
end = None
try:
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) == 6 and p[1] == "range" and p[2] == typ and p[3] == order:
            end = max(end or 0, int(p[5]))
except FileNotFoundError:
    pass
print(end if end is not None else default)
EOF
)
bin=main-$([ "$UNBOUND" = 1 ] && echo u)$([ "$LARGE_BIOMES" = 1 ] && echo lb || echo sb)
# Build under the lock that tools/bench.sh also takes, and run a private copy, so a benchmark
# build in the same tree can never replace the binary the search runs.
{
  flock 9
  make -s NO_NET=1 SEED_ORDER="$SEED_ORDER" LARGE_BIOMES="$LARGE_BIOMES" UNBOUND="$UNBOUND" PRINT_INTERVAL=64 MAX_ITERS=0 -B >/dev/null
  cp "$bin" search/search-bin
} 9>/tmp/commission-build.lock
echo "search $TYPE $SEED_ORDER from logical $next"
./search/search-bin --start "$next" --pipelines "$P" --threads 4 --output /dev/null "$@" 2>&1 \
  | python3 tools/search_log.py search "$TYPE" "$SEED_ORDER" "$next" "$P" "$BATCH"
