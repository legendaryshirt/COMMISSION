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
| gv1 256-row masked + `__launch_bounds__(256,4)` (≤64 regs, 4 blocks/SM) | — | 2672 | — | 6805 | 20.20 | rejected (+18% gv1 vs 2261 same-session ref) |
| gv1 256-row masked + minBlocks 5 / 6 | — | 5690 / 16768 | — | 9820 / 20886 | 13.99 / 6.58 | rejected: register spills. gv1 isn't latency-bound, so occupancy doesn't help |
| f2_01a/b: linear-in-x cell evaluation (F2_LINX) | — | — | 1444 | 6393 | 21.50 | rejected: no gain, count drift of 1. Removes ALU only, and the kernel is LSU-bound |
| **gradDot packed table: 1 shared load per corner instead of 3 (byte-encoded ±2.0)** | — | — | **1115** (ref 1427) | **6053** (ref 6365) | **22.71** | **kept: 2_01a −22%, 2_01b −20%, 2a −22%, total −4.9%, all counts byte-identical** |
| gv1 warp-uniform gate skip: per-64-row-window max of each gate term (exact, counts identical) | — | 5630 (ref 2257) | — | 9414 | 14.60 | **rejected, 2.5x slower**: the bound passes 99.999% of candidates |
| **f2_01a/b: byte-pair permutation table, 7 shared loads per cell instead of 14** | — | — | **1021** (ref 1115) | **5958** (ref 6053) | **23.07** | **kept: 2_01a −8.4%, 2_01b −8%, total −1.6%, counts byte-identical** |
| filter_seeds: U32 float convert + skip zero-state fix-up (FS_FAST), golden | 1873 (ref 1932) | — | — | 5738 (ref 5799) | **23.95** | **kept: seeds −3.0%, total −1.0%, counts identical** |
| **filter_seeds: per-warp shared queue compaction of first-test survivors (FS_COMPACT), chain 16** | 1754 (ref 1878) | — | — | 5622 (ref 5755) | 24.45 | **kept: seeds −6.6%, counts identical** |
| FS_COMPACT, chain 32 / 64 / 128 / 256 | 1715 / 1695 / 1690 / 1698 | — | — | 5586 / 5559 / 5555 / 5562 | 24.60 / **24.72** / 24.74 / 24.71 | **chain 64 is the default** (plateau) |
| gv1: gate pass bits as a predicate mask + one warp `any` (GV1_MASK) | — | 2246 (ref 2258) | — | 5550 (ref 5559) | 24.77 | kept: −0.5% gv1, exact |
| gv1: uint32 pre-scaled gate offsets via broadcast LDS.128 (GV1_IDX32) | — | 2926 (ref 2246) | — | 6219 | 22.10 | **rejected, +30% gv1** |
| f2: pair table extended with gradient words (uint2 per entry, 7 LDS per cell instead of 15; F2_G2) | — | — | 1134 (ref 1023) | 5667 (ref 5547) | 24.25 | **rejected, +11% 2_01a**: build cost (9 LDS + 4 STS.128 per lane per octave) and 5 KB/warp smem (33% occupancy) outweigh it |
| gv1: 2 z-rows per thread (index unpacked once, 2nd row = +immediate), GV1_ZPT=2 | — | 2191 (ref 2254) | — | 5501 | 24.98 | kept (ZPT=4: 2857, rejected; occupancy 6 warps/SM) |
| gv1: full-score columns single 256-row copy + mask, conv-kernel table via L1 (GV1_SLIM): smem 30 → 17 KB, 5 blocks/SM | — | 2105 | — | 5418 | 25.37 | kept |
| gv1: next 8 index bytes per row as one 8-byte load | — | 2002 | — | 5314 | 25.86 | kept |
| gv1: conv-kernel table re-laid out for 16-byte loads | — | 2012 | — | 5323 | 25.82 | rejected (same wavefronts) |
| gv1: carry gate entry across groups (8 loads/8 cand) + `__launch_bounds__(128,5)` | — | **1878** | — | **5185** | **26.51** | **kept** (without the bound: 1991, regs > 102 cost a block) |
| filter_seeds: queue first-test state (seed, RNG after fork, a-yo fork, score) so the dense pass resumes instead of recomputing ~85 ALU ops/survivor | **1584** (ref 1699) | 1880 | — | **5075** | **27.08** | **kept: −6.8% seeds, counts identical** |
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

Note (2026-11-06): the unchanged gv1 512-row config measured 2261 ms in this session, against 2317
on 2026-11-05: a 2.4% clock/thermal drift between sessions. Only compare against a same-session
reference row.

## Why filter_2_01a is LSU-bound (ncu, 2026-11-06)
LSU pipe 98.9%, issue 65%. Ada issues about 0.5 LSU warp-instructions per clock per SM, so
the ~290 shared loads per warp-input outweigh ~1590 total instructions / 4 schedulers. nvcc had
already hoisted the gradient-table loads into the cell-change branch: dynamic shared loads were
216.3 M with or without F2_LINX. So only fewer loads per cell help. The packed table cuts each
corner from 3 loads to 1 and rebuilds the components with `__byte_perm`. It stores 2*g, which is
exact; callers multiply the octave weight by 0.5, which is also exact.
Remaining LSU per cell after the pair table: 7 q[] loads + 8 gradient-table loads. Folding gradients
into a per-warp pair table would cost ~41 KB smem/block (2 blocks/SM); not tried.

## gv1 gate selectivity (GV1_STATS instrumentation, 2026-11-06)
Only 0.174% of lane-candidates pass the -12 gate, and 4.9% of warp-candidates (32 z-lanes) have
any passing lane. But a warp-uniform upper bound built from per-term window maxima passes almost
always: each term's max over 32-64 rows is high, and selectivity comes from the 4 terms being high
together on the same lane. Any skip scheme has to bound the joint sum, not the terms separately.

## filter_seeds roofline (2026-11-06)
Ada's ALU pipe (LOP3/SHF/IADD3) does 64 lanes/clk/SM. The SASS has about 150 ALU-pipe ops per seed
(84 LOP3, 44 SHF.L.W rotates, ...), which is about 64/clk/SM at the measured 73 G seeds/s. It sits
on that roofline, so only removing ops helps.

## Cumulative (2026-11-06, golden default): 20.88 -> 27.08 G seeds/s (+29.7%) at identical funnel output.

## Why FS_COMPACT works
With 21.7% passing the first test, P(some lane of 32 passes) = 99.96%, so under per-lane branching
every lane paid for the second-stage forks. Survivors are now queued per warp in shared memory and
re-run 32 at a time; the dense pass repeats the cheap first test and recomputes the 2 mix64s. A
longer chain amortizes the flush of the partial queue at the end.

## Where things stood before the speed-of-light pass (2026-11-06, superseded below)
All three hot kernels are at a measured hardware limit for their current algorithms:
filter_seeds on ALU-pipe integer ops, gv1 on issue/LSU (balanced), filter_2_01a on LSU issue
(balanced against ALU issue). Every remaining idea tried costs as much as it saves. Further big
gains would need a different algorithm (e.g. a cheaper exact pre-test in front of gv1), not tuning.
