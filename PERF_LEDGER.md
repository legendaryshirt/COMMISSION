# Performance ledger (perf branch)

Hardware: compute node, RTX 4070 Ti (sm_89, 60 SMs), CUDA 13.4.
Benchmark: `tools/bench.sh 512 1000000000000` (512 iterations x 2^28 seeds, small biomes,
bounded). Run-to-run noise is about ±0.2%. Every change must leave the funnel counts
byte-identical:
gv1 386342512, 2_01a 1502284, gv2 13412604, 2_01b 412844, 2a 26736, 2b 2028, 2c 66, 2d 1.

| change | filter_seeds ms | gv1 ms | 2_01a ms | total ms | G seeds/s | verdict |
|---|--:|--:|--:|--:|--:|---|
| upstream main d5a0179 | 2093 | 2468 | 1427 | 6583 | 20.88 | baseline |
| gv1: column-major conv tables + packed float2 rolling gate (GV1_ROWS=512) | 2094 | 2317 | 1427 | 6436 | 21.36 | **kept, −6.1% gv1, +2.3% total** |
| gv1 same, single 256-row table + index mask | 2096 | 2705 | 1428 | 6825 | 20.14 | rejected (ALU-bound: the mask costs more than the smem it saves) |
| `SEED_ORDER=golden` straight reference (2 mix64/seed, same seeds) | 2094 | 2324 | 1431 | 6449 | 21.31 | reference for the next row |
| `SEED_ORDER=golden`, chain 8 (1 mix64/seed) | 1942 | 2322 | 1430 | 6293 | 21.84 | counts identical to reference |
| `SEED_ORDER=golden`, chain 16 | 1925 | — | — | 6274 | 21.91 | **kept, now the DEFAULT (2026-11-05): −8.1% seeds, +2.6% total** (4: 6324, 32: 6270) |
| gradDot as branchless ALU instead of 3 shared-table loads | 2098 | 2321 | 2238 | 7318 | 18.78 | **rejected, 2_01a +57%** |

## Profile (ncu, baseline + gv1 change)
- filter_seeds: SM 95% busy, 23 regs, 100% theoretical occupancy. 64-bit integer ALU bound.
- gv1: issue 76%, 80 regs + 30 KB smem/block -> 3 blocks/SM = 50% occupancy.
- filter_2_01a: issue 65%, ncu "memory throughput" 99%, 80 regs, 50% occupancy. Despite that
  figure, moving gradDot from shared memory to the ALU lost 57%, so the ALU is what's scarce
  on Ada. Shared-table lookups are cheap here.

## SEED_ORDER=golden (default; `make SEED_ORDER=linear` restores upstream order)
Logical index n maps to seed (n·GOLDEN) ^ SILVER. That's a bijection on 64-bit space because
GOLDEN is odd, so coverage stays complete and non-overlapping. The point is that seed n's
`hi` mix64 equals seed n+1's `lo` mix64, so a thread walking 16 consecutive n pays 17 mixes
instead of 32. Golden-order funnel counts differ from linear because these are different seeds.
Correctness check: the chained kernel and a straight 2-mix kernel over the same seeds give
identical counts at every stage (21870676 / 386426648 / 1503064 / 13432243 / 414115 / 2216 / 72 / 3).
End to end, a 120 s run with `--size 2000000` produced 32 cubiomes-confirmed islands.
Cost of the change: a given `--start` covers different seeds than upstream, and printed seeds are
full 64-bit values. The net handshake becomes `OMISSION-GPZ` so mixed builds can't share a server.
The gain is only +2.6% because mix64 is about 30% of filter_seeds; the fork/rotate chain is the rest.

## Upstream warts noticed (not fixed yet)
- The same island is emitted 2-3x (neighbouring GPU candidates hit it), so CPU verification
  is repeated and output.txt gets duplicate lines. A host-side dedup ring before the CPU queue
  would fix this.

## Pending job
`cnode job commission-order-stats` (started 2026-11-05): linear vs golden, 16384 iterations each (~4.4 T seeds), stage pass rates. Collect with `cnode pull order-stats.txt`; the 1-in-N columns should agree within noise.
Result (4.4 T seeds each): stage pass rates agree to 4 significant figures (filter_seeds 1 in 6285.29
vs 6285.05, gv1 3708.97 vs 3708.97, 2_01a 257.02 vs 257.14). Finals 65 vs 49 is a 1.5 sigma Poisson
difference, so not significant. Golden total 21.81 vs 21.27 G seeds/s (+2.5%).

## Live search
`cnode job commission-search 'tools/search.sh'` (started 2026-11-05 ~23:30Z). It's golden,
small biomes, bounded, min size 10M, logical start 2^62. On the node: islands go to
`search/current.found`, covered ranges to `search/ranges.tsv` once a session closes. To stop:
`cnode stop commission-search`, then `ssh compute 'cd COMMISSION && tools/search.sh close'`
to log the range. Or just restart it, since the next run closes the previous session itself.
Collect with `cnode pull search`.

## Host-side candidate dedup (2026-11-05)
Before a candidate is queued for CPU verification, it's dropped if one of the last 64 queued
candidates has the same seed and lies within 1024 blocks in both x and z. The CPU window is ±12800.
Check: a 120 s run at --size 2000000 printed 12 lines / 12 unique islands, against 32 lines /
23 unique before. 4 candidates were dropped.
**Fixed-work check passed (2026-11-06, job commission-dedup-check):** 4.4 T identical seeds
from 1e12, --size 2000000, CPU queue fully drained (MAX_ITERS builds now exit through main's
graceful SIGINT path). Without dedup: 48 lines / 37 unique islands. With dedup radius 1024: 37 lines /
37 unique, an identical set (diff empty). So no island is lost, and 11 redundant CPU verifications are gone.
Earlier note, kept for history: a fixed-work check was owed. Time-limited runs verify a different subset (the single CPU
thread is the bottleneck), so the unique counts above aren't comparable. Run identical seeds with
the CPU queue fully drained, with and without dedup, and confirm the unique island sets match.
