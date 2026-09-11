# Optimization Log

## How to read this log

This file preserves the engineering decisions behind M0–M5. Each experiment
records its baseline, evidence, hypothesis, implementation, measurement,
interpretation, and decision. Earlier backends remain in the repository, so a
later result does not erase the comparison point that motivated it.

Unless stated otherwise, benchmark tables use a Release build, normalized
deterministic FP32 data, `Q=4`, `K=10`, seed `42`, two warm-ups, and
five measured iterations. Kernel latency is measured with CUDA Events.
End-to-end CUDA latency includes allocation, H2D, the kernel, score D2H, CPU
Top-K, and cleanup. Dataset generation is excluded.

The validation environment was WSL2 Ubuntu 22.04.4, RTX 3050 Ti Laptop GPU
(4 GB, compute capability 8.6), driver 576.88, CUDA/NVCC 12.9/12.9.86,
GCC 11.4.0, CMake 3.22.1, Nsight Systems 2025.1.3, and Nsight Compute
2025.2.1.

Results from separate process runs show normal variation. In particular, the
M4 primary and dimension-sweep D=768 rows are distinct measurements, and M5
observed much slower host-to-device transfer than M2/M4. Raw values are
retained rather than adjusted to make tables agree.

## M0 — CPU correctness reference

### Baseline

No search implementation existed.

### Evidence / observation

All later CUDA experiments needed a deterministic oracle and a backend-neutral
request/result contract. Exact Top-K also needed a stable rule for equal
scores.

### Hypothesis

A simple serial implementation would make correctness failures attributable
before parallel execution or floating-point reduction order complicated the
system.

### Implementation

M0 added deterministic synthetic FP32 data, row normalization, serial dot
products, deterministic CPU Top-K (score descending, index ascending), the
`SearchBackend` interface, CLI, benchmark harness, and CPU tests.

### Measurement

The historical M1 validation run measured the same CPU implementation:

| Workload | CPU E2E |
| --- | ---: |
| N=10,000, D=128 | 1.952 ms |
| N=100,000, D=128 | 21.565 ms |
| N=100,000, D=768 | 170.975 ms |

### Interpretation

These values establish a serial reference on one host, not a claim that the
CPU backend is optimized.

### Decision

Keep the CPU path unchanged as the correctness oracle. Every CUDA backend must
match indices and scores within a documented FP32 tolerance.

## M1 — Naive CUDA baseline

### Baseline

The M0 CPU backend computed every query/vector dot product serially.

### Evidence / observation

A deliberately simple CUDA version was required before profiling or
optimization. There was no evidence yet for choosing a more complex mapping.

### Hypothesis

Mapping one query/vector pair to one CUDA thread would provide a correct,
readable CUDA baseline and expose the costs of kernel execution, data
movement, and per-search resource management.

### Implementation

`cuda-naive` launches 256 threads per block over `Q*N` pairs. Each thread
loops over all `D` dimensions and writes one score. The host allocates
database/query/score buffers per search, performs synchronous copies, copies
all scores back, and reuses CPU Top-K. All CUDA operations and launches are
checked.

### Measurement

Historical M1 averages:

| Workload | CPU E2E | Naive kernel | Naive E2E | CPU / CUDA E2E |
| --- | ---: | ---: | ---: | ---: |
| N=10,000, D=128 | 1.952 ms | 0.193 ms | 2.186 ms | 0.893x |
| N=100,000, D=128 | 21.565 ms | 3.688 ms | 16.598 ms | 1.299x |
| N=100,000, D=768 | 170.975 ms | 65.644 ms | 117.316 ms | 1.457x |

### Interpretation

The baseline was correct and faster than the serial CPU for the two 100K
workloads, but slower at 10K. No causal performance claim was made because no
stage or profiler evidence existed.

### Decision

Freeze `cuda-naive` as the M1 algorithmic baseline. Investigate before
changing the kernel.

## M2 — Latency decomposition

### Baseline

The unchanged M1 pipeline reported one host E2E number and one CUDA Event
kernel number.

### Evidence / observation

At small workloads, a sub-millisecond kernel could coexist with CUDA E2E slower
than CPU. At high `D`, both the kernel and database byte volume grew. The
aggregate measurements could not identify which stage controlled the result.

### Hypothesis

Separating allocation, copies, kernel, CPU Top-K, cleanup, and total E2E would
identify the largest measured opportunities without altering the baseline.

### Implementation

M2 added opt-in `--timing-breakdown`. CUDA Events measure synchronous H2D,
kernel, and D2H operations; host `steady_clock` measures allocations, CPU
Top-K, cleanup, and E2E. The kernel, mapping, block size, score transfer, and
Top-K algorithm were unchanged.

### Measurement

From [`m2_stage_timing.csv`](../benchmarks/results/m2_stage_timing.csv):

| Workload | Allocation | DB H2D | Kernel | Score D2H | CPU Top-K | Cleanup | Total E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10K x 128 | 0.481 | 0.613 | 0.195 | 0.074 | 0.275 | 0.361 | 2.246 ms |
| 100K x 128 | 0.837 | 4.534 | 3.523 | 0.287 | 2.481 | 0.656 | 13.885 ms |
| 100K x 256 | 1.320 | 10.860 | 18.909 | 0.486 | 2.202 | 1.057 | 37.760 ms |
| 100K x 768 | 1.694 | 31.888 | 64.144 | 0.612 | 2.398 | 2.149 | 110.006 ms |

The component sum is diagnostic and does not include every metadata call,
event creation, or host container operation. It is not forced to equal E2E.

The uninstrumented crossover sweep placed the observed CPU/CUDA boundary
between 10K and 15K vectors at D=128 on this host. That boundary is
noise-sensitive, not universal.

### Interpretation

At 100K x 768, the kernel was the largest measured stage (58.3% of E2E) and
database H2D was next (29.0%). At 10K x 128, allocation plus cleanup was
0.842 ms while the kernel was only 0.195 ms, explaining why kernel speed alone
did not determine E2E.

The full score D2H copy was only 0.612 ms at 100K x 768, about 0.6% of E2E,
so GPU Top-K was not the highest-evidence first target.

### Decision

Profile the high-D similarity kernel to determine why it was slow. Retain
database residency as a later, separate system-level opportunity.

## M2.5 — Nsight Compute diagnosis

### Baseline

The frozen `cuda-naive` kernel used one thread per pair, a sequential
dimension loop, row-major data, and a 256-thread block.

### Profiler evidence

The focused primary profile used `N=100,000,D=768,Q=4,K=10`. Nsight Compute
reported:

| Metric | D=768 naive | D=128 naive |
| --- | ---: | ---: |
| Useful global-load bytes / 32-byte sector | 4 | 4 |
| DRAM rate | 47.81 GB/s | 102.24 GB/s |
| Compute throughput | 0.26% | 6.39% |
| No-eligible-warp cycles | 97.99% | 96.58% |
| Eligible warps / scheduler | 0.05 | 0.07 |
| Warp cycles / issued instruction | 551.48 | 301.21 |
| Achieved occupancy | 95.43% | 85.67% |
| Executed instructions | 77,612,532 | 13,612,532 |

At D=768, dominant stalls were approximately 60.46% LG-memory queue throttle
and 39.71% L1TEX scoreboard dependency. High occupancy coexisted with almost
no eligible warps.

### Hypothesis

Adjacent naive threads load the same dimension from different row-major
vectors. At D=768 their addresses are 3,072 bytes apart, so each thread uses
only one 4-byte word from a 32-byte sector. Assigning cooperating threads to
adjacent dimensions should improve transaction use and expose dimension-level
parallelism.

The sequential-D hypothesis was only partially supported: instruction count
and latency grew with D, but memory stalls—not a measured scalar dependency—
dominated.

### Implementation

None. M2.5 was analysis-only. It generated compact Nsight Compute CSV exports
and selected the next controlled experiment.

### Measurement

An unprofiled validation run measured 60.130 ms kernel / 106.778 ms E2E at
D=768 and 3.493 ms / 14.080 ms at D=128. Nsight replay duration was kept
separate from benchmark timing.

### Interpretation

The 4/32 result and stride warning confirmed inefficient global-load
transactions. Low compute and sub-peak DRAM throughput did not support calling
the kernel purely compute-bound or bandwidth-saturated; latency exposure was
the more accurate description.

### Decision

Implement one separate block-per-vector experiment. Preserve the naive backend
and all host-pipeline behavior.

## M3 — Block-per-vector reduction

### Baseline

The primary M2/M2.5 target was a roughly 64 ms D=768 naive kernel with 4/32
useful load bytes, almost no eligible warps, and heavy memory stalls.

### Profiler evidence

The naive mapping made neighboring threads walk different database rows. The
profile supported changing the thread/data mapping, not changing storage
precision, Top-K, or the host lifecycle.

### Hypothesis

One 256-thread block per pair, with threads walking dimensions
`t, t+256, ...`, would coalesce global loads and parallelize D. A
shared-memory tree reduction would produce the score. The reduction could be
too expensive for small D.

### Implementation

M3 added `cuda-block` as a new backend. It kept per-search allocations,
synchronous copies, full score D2H, CPU Top-K, timing boundaries, FP32 data,
and the fixed 256-thread block. It added a 256-float shared reduction array
and block-wide barriers.

### Measurement

From the M3 controlled runs:

| Workload | Naive kernel | Block kernel | Kernel speedup | Naive E2E | Block E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary 100K x 768 | 63.835 | 7.993 | 7.986x | 111.474 | 56.139 ms |
| 100K x 128 | 3.544 | 5.476 | 0.647x | 13.969 | 16.459 ms |
| 100K x 256 | 18.923 | 5.637 | 3.357x | 37.082 | 23.640 ms |
| Dimension-run 100K x 768 | 63.524 | 7.994 | 7.946x | 111.358 | 52.090 ms |
| Small 10K x 128 | 0.184 | 0.539 | 0.341x | 1.864 | 2.549 ms |

The paired D=768 profiler comparison measured:

| Metric | Naive | Block |
| --- | ---: | ---: |
| Useful global-load bytes / sector | 4 | 32 |
| DRAM rate | 51.41 | 125.76 GB/s |
| No-eligible cycles | 98.67% | 41.01% |
| Eligible warps / scheduler | 0.03 | 1.31 |
| Direct LG-throttle share | 63.47% | 0.00% |
| Warp cycles / issued instruction | 860.95 | 19.02 |
| Achieved occupancy | 95.46% | 93.93% |
| Static shared memory / block | 0 | 1.02 KiB |

### Interpretation

The experiment confirmed its primary mechanism: load-sector use improved
eightfold and high-D kernel latency improved about eightfold. Occupancy
remained high while eligible warps increased.

The tradeoff was also measured. At D=128, all 256 threads still participated
in shared-memory reduction and barriers even though only 128 dimensions
existed. Global score stores used 4/32 bytes per sector because one thread per
block wrote the result, and L2 hit rate decreased; neither was the primary
load target.

### Decision

Keep `cuda-block` as a successful high-D experimental backend. Test a
warp-sized reduction to retain coalescing while reducing low-D overhead.

## M4 — Warp-per-vector reduction

### Baseline

M3 achieved 32/32 load-sector use and about an 8x D=768 kernel improvement,
but regressed at D=128 because of the fixed full-block reduction.

### Profiler evidence

At D=128, block-per-vector executed 444,000,000 instructions in the focused
profile and spent 6.6 cycles per warp waiting at a CTA barrier (33.9% of its
19.5 cycles per issued instruction).

### Hypothesis

Mapping one pair to one warp would preserve adjacent-dimension loads while
using only 32 cooperating lanes and a five-step shuffle reduction. This should
remove shared reduction state and block-wide barriers, especially helping
smaller dimensions, without assuming a win at every D.

### Implementation

M4 added `cuda-warp`. A 256-thread block contains eight independent pair
warps. Each lane walks `l, l+32, ...`, `__shfl_down_sync` reduces at
offsets 16/8/4/2/1, and lane 0 writes the score. A ballot-derived mask handles
tail warps. The host pipeline stayed frozen.

### Measurement

The primary run:

| Workload | Naive kernel | Block kernel | Warp kernel | Warp vs naive | Warp E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100K x 768 | 64.447 | 7.879 | 6.918 | 9.316x | 50.381 ms |

Representative rows from the independent dimension sweep:

| D | Naive kernel | Block kernel | Warp kernel | Warp vs naive |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 3.388 | 5.712 | 1.123 | 3.017x |
| 256 | 19.149 | 5.559 | 2.351 | 8.145x |
| 768 | 63.601 | 7.975 | 6.985 | 9.105x |

At D=128, the focused block/warp profile changed executed instructions from
444,000,000 to 80,400,000, no-eligible cycles from 44.07% to 38.85%, and warp
cycles per issued instruction from 19.45 to 15.91. Both kernels retained
32/32 useful global-load bytes.

At D=768, warp reached 177.48 GB/s, used 38 registers/thread, achieved 89.31%
occupancy, and had 73.72% no-eligible cycles. It was faster in the normal
kernel benchmark despite worse scheduler eligibility than block.

### Interpretation

The warp mapping removed the measured low-D barrier/instruction overhead and
kept the coalesced loads. The D=768 result was competitive rather than
universally dominant: block E2E was 51.193 ms and warp E2E was 51.987 ms in
the dimension sweep, while warp won 50.381 versus 51.515 ms in the separate
primary run. The approximately 1.5% reversal is normal run-to-run variance,
not a contradiction or a reason to rewrite raw data.

Naive remained the fastest kernel at D=32. M4 therefore demonstrates a useful
mapping, not an automatic selection rule.

### Decision

Keep all three CUDA kernel designs. Use the stable primary 9.316x
warp-versus-naive kernel result as the headline. Defer warps-per-block tuning;
the already measured request-level database H2D cost motivates a distinct
system experiment.

## M5 — GPU-resident database

### Baseline

The M4 primary kernel was about 6.9 ms but stateless E2E was about 50 ms.
Earlier M2 decomposition measured database H2D at 31.888 ms for 100K x 768.
Every stateless search allocated and uploaded the same database again.

### Profiler evidence

The stateless pipeline's allocation/copy/free sequence was visible in Nsight
Systems API traces. M2 timing had already established database H2D as the
largest non-kernel stage at high D.

### Hypothesis

For repeated query batches against an unchanged database, allocating and
uploading the database once should reduce steady-state latency. A one-shot
request should not improve because preparation must still be paid.

### Implementation

M5 added `ResidentSearchBackend` and `cuda-warp-resident` with explicit
`prepare_database`, `reload_database`, `search`, and `clear_database`
states. The M4 warp mapping is unchanged. Database allocation/H2D becomes
preparation work; query/score allocation, query H2D, kernel, score D2H, CPU
Top-K, and query cleanup remain per search.

### Measurement

Dedicated one-shot measurements:

| Workload | Stateless E2E | Resident prepare | First query | Resident cold total | Cold ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary 100K x 768 | 526.642 | 660.766 | 15.873 | 676.639 ms | 0.778x |
| 100K x 128 | 168.031 | 362.981 | 9.331 | 372.313 ms | 0.451x |
| 100K x 256 | 233.638 | 432.986 | 9.934 | 442.921 ms | 0.527x |
| Dimension-run 100K x 768 | 496.184 | 683.011 | 15.672 | 698.683 ms | 0.710x |

Cold ratio is stateless/resident; values below 1 are regressions.

Warm means over 100 measured query batches after two warm-ups:

| Workload | Stateless mean | Resident mean | Same-run warm speedup |
| --- | ---: | ---: | ---: |
| Primary 100K x 768 | 395.504 | 13.289 | 29.761x |
| 100K x 128 | 72.177 | 6.986 | 10.331x |
| 100K x 256 | 136.776 | 8.278 | 16.523x |
| Dimension-run 100K x 768 | 394.634 | 13.437 | 29.369x |
| Optional 250K x 768 | 986.258 | 33.958 | 29.043x |

Primary per-batch stage means:

| Stage | Stateless | Resident warm |
| --- | ---: | ---: |
| Database H2D | 369.096 | 0 ms |
| Kernel | 7.459 | 7.038 ms |
| Score D2H | 2.321 | 2.136 ms |
| CPU Top-K | 2.415 | 2.452 ms |
| Stage E2E | 395.502 | 13.287 ms |

The M5 stateless transfer timing was much slower than the earlier M2/M4 runs.
The paired result is retained as environment-sensitive evidence; it is not
used as the project's headline.

Amortization for the primary run crossed the measured table at two query
batches: one batch was 0.577x stateless/resident, two were 1.130x, and 100
were 19.760x after including preparation.

Nsight Systems recorded nine `cudaMemcpy` calls for three stateless searches
and seven for resident: three database H2D + three query H2D + three score D2H
versus one database H2D + three query H2D + three score D2H. Kernel launches
remained three for both.

### Interpretation

The unchanged kernel means (7.459 versus 7.038 ms) and API counts confirm a
system-lifecycle change rather than a new kernel optimization. Warm repeated
searches benefited because database H2D disappeared from each request. Cold
resident searches regressed exactly as predicted.

After residency, the remaining bottleneck became workload-dependent. At
D=768 the 7.038 ms kernel was about 53% of 13.289 ms query E2E; at D=128 the
2.481 ms CPU Top-K stage exceeded the 1.158 ms kernel.

### Decision

Keep `cuda-warp-resident` for repeated queries against an unchanged
database. Report cold, warm, and amortized timing separately. Do not present
the environment-sensitive roughly 30x warm ratio as a universal or one-shot
speedup.

## M5.5 — Project finalization

### Baseline

M0–M5 were implemented and measured, but the top-level narrative, artifact
organization, and reproduction path were spread across milestone-era files.

### Evidence / observation

The final architecture document still contained future plans and obsolete
paths; M5 emitted two byte-for-byte duplicate CSVs; large local profiler
binaries needed an explicit policy; and the repository had no final
source-backed performance figures or interview notes.

### Hypothesis

Curating documentation and artifacts around the validated evidence would make
the work reviewable and reproducible without changing performance behavior.

### Implementation

M5.5 rewrote the README and final architecture, reconciled this log and the
performance report, added benchmark/profiling catalogs, removed only the two
duplicate M5 CSV copies, added a reproducible matplotlib figure script and two
figures, expanded ignore rules, and added resume/interview notes.

### Measurement

No new performance benchmark was introduced. The figures read the existing
M4 dimension CSV; all numeric claims trace to retained raw benchmark or
profiler exports.

### Interpretation

This is release/polish work. A successful clean build/test demonstrates
reproducibility but is not a new optimization result.

### Decision

Stop at M5.5. All later ideas remain explicitly labeled future work.
