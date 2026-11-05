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
| gradDot as branchless ALU instead of 3 shared-table loads | 2098 | 2321 | 2238 | 7318 | 18.78 | **rejected, 2_01a +57%** |

## Profile (ncu, baseline + gv1 change)
- filter_seeds: SM 95% busy, 23 regs, 100% theoretical occupancy. 64-bit integer ALU bound.
- gv1: issue 76%, 80 regs + 30 KB smem/block -> 3 blocks/SM = 50% occupancy.
- filter_2_01a: issue 65%, ncu "memory throughput" 99%, 80 regs, 50% occupancy. Despite that
  figure, moving gradDot from shared memory to the ALU lost 57%, so the ALU is what's scarce
  on Ada. Shared-table lookups are cheap here.
