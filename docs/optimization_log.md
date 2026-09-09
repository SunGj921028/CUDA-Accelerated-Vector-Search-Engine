# Optimization Log

This log preserves the measured progression from a simple baseline to later
profile-driven redesigns. M1 is a baseline implementation, not an optimized
CUDA kernel.

# M1 — Naive CUDA Baseline

## Kernel Design

One CUDA thread computes one complete query/database-vector dot product. For
`Q` queries and `N` database vectors, the grid covers `Q * N` pairs. Each
thread sequentially processes all `D` dimensions and writes one FP32 score to
global memory. The implementation uses a fixed 256 threads per block.

## Motivation

Establish the simplest correct CUDA implementation before optimization. The
CPU backend remains the correctness oracle, and the full score array is copied
back to the CPU so M1 exposes a clear end-to-end baseline for later profiling.

## Implementation Change

Added the `cuda-naive` `SearchBackend` implementation and kept the existing
CPU backend and CPU Top-K selection unchanged. The CUDA-enabled CMake option
compiles only the CUDA source and exposes the backend through the existing CLI
factory.

## Baseline Measurements

Validation configuration: Ubuntu 22.04.4 under WSL2, NVIDIA GeForce RTX 3050
Ti Laptop GPU, 4 GB VRAM, driver 576.88, CUDA Toolkit 12.9, NVCC 12.9.86,
GCC 11.4.0, Release build, `Q = 4`, `K = 10`, seed `42`, two warm-ups, and
five measured iterations. Latencies are average values in milliseconds. CUDA
kernel latency is measured with CUDA Events; CUDA end-to-end latency includes
per-search allocation, H2D copies, kernel, D2H score copy, CPU Top-K, and
cleanup.

| Workload | CPU latency | CUDA kernel-only | CUDA end-to-end | CPU / CUDA end-to-end | CPU queries/s | CUDA queries/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `N=10,000, D=128` | 1.952 ms | 0.193 ms | 2.186 ms | 0.893x | 2,048.983 | 1,829.783 |
| `N=100,000, D=128` | 21.565 ms | 3.688 ms | 16.598 ms | 1.299x | 185.488 | 240.997 |
| `N=100,000, D=768` | 170.975 ms | 65.644 ms | 117.316 ms | 1.457x | 23.395 | 34.096 |

These are baseline observations from the measured runs, not claims about why
the timings differ. CPU and CUDA used the same deterministic dataset
configuration. Result checksums were within the expected FP32 accumulation
difference, and the correctness tests compared ranked IDs and scores with a
tolerance.

## Interpretation

The measurements establish a reproducible M1 reference across small and
larger workloads. No profiler evidence was collected in M1, so the results do
not identify the cause of any timing difference.

## Expected Weaknesses

These are hypotheses for later profiling, not confirmed bottlenecks:

- sequential dimension work within each CUDA thread;
- potentially inefficient global-memory access patterns;
- complete GPU-to-CPU transfer of the similarity scores;
- CPU-side Top-K selection;
- CUDA launch and transfer overhead on small workloads; and
- utilization that may depend strongly on workload shape.

## Decision

Keep this backend unchanged as the M1 reference. Profiling and any redesign
belong to later milestones; no M2 optimization is included here.

# M2 — Profiling and Bottleneck Analysis

M2 is an analysis and instrumentation milestone. The `naive_similarity_kernel`
body, one-thread-per-query/vector mapping, sequential dimension loop, fixed
256-thread block, full score D2H copy, and CPU Top-K are unchanged from M1.
No M3 optimization was implemented.

## Baseline

The M1 reference measurements remain the historical baseline and are not
rewritten here:

| Workload | CPU E2E | CUDA kernel | CUDA E2E | CPU/CUDA E2E |
| --- | ---: | ---: | ---: | ---: |
| `N=10,000, D=128` | 1.952 ms | 0.193 ms | 2.186 ms | 0.893x |
| `N=100,000, D=128` | 21.565 ms | 3.688 ms | 16.598 ms | 1.299x |
| `N=100,000, D=768` | 170.975 ms | 65.644 ms | 117.316 ms | 1.457x |

The M2 runs below were collected later on the same WSL2 host and are
measurement records for analysis, not replacements for the M1 table. They use
Release builds, `Q=4`, `K=10`, seed `42`, two warm-ups, and five measured
iterations. Dataset generation is excluded from backend latency.

## Stage Timing

`--timing-breakdown` enables the additional instrumentation. It is disabled by
default so ordinary M1 benchmark runs do not acquire the extra event and
synchronization overhead.

The timing scopes are:

* `allocation_ms`: host `steady_clock` around the three `cudaMalloc` calls.
* `h2d_database_ms`, `h2d_queries_ms`, and `d2h_scores_ms`: CUDA Event pairs
  around the existing synchronous `cudaMemcpy` calls; the stop event is
  synchronized before reading elapsed time.
* `kernel_ms`: the existing CUDA Event measurement around the unchanged
  similarity kernel.
* `cpu_topk_ms`: host `steady_clock` around host score slicing and the existing
  CPU Top-K loop.
* `cleanup_ms`: host `steady_clock` around event destruction and the three
  explicit `cudaFree` calls.
* `total_e2e_ms`: host `steady_clock` from after request validation through
  device selection/properties, allocation, transfers, kernel synchronization,
  CPU Top-K, and cleanup.

The CUDA copies are synchronous in M1, so the event/synchronization boundaries
make their measurements meaningful. The component sum is deliberately not
forced to equal E2E. Device metadata queries, event creation, host score and
result allocations, API overhead, and the different timing boundaries account
for the residual shown below.

All values are average milliseconds from
`benchmarks/results/m2_stage_timing.csv`:

| Workload | Allocation | H2D DB | H2D Query | Kernel | D2H Scores | CPU Top-K | Cleanup | Component sum | Residual to E2E | Total E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `N=10,000, D=128` | 0.481 | 0.613 | 0.058 | 0.195 | 0.074 | 0.275 | 0.361 | 2.057 | 0.189 | 2.246 |
| `N=100,000, D=128` | 0.837 | 4.534 | 0.129 | 3.523 | 0.287 | 2.481 | 0.656 | 12.447 | 1.438 | 13.885 |
| `N=100,000, D=256` | 1.320 | 10.860 | 0.130 | 18.909 | 0.486 | 2.202 | 1.057 | 34.964 | 2.796 | 37.760 |
| `N=100,000, D=768` | 1.694 | 31.888 | 0.116 | 64.144 | 0.612 | 2.398 | 2.149 | 103.001 | 7.005 | 110.006 |

The instrumented backend E2E values were 2.247, 13.886, 37.762, and 110.008
ms for those rows. They are close to the backend timer, while the extra
instrumentation makes the 10K case about 0.2 ms slower than the uninstrumented
M1-style run. This is why stage results are reported separately from the
normal crossover and dimension sweeps.

At `N=10,000, D=128`, the measured kernel is only 0.195 ms, or 8.7% of the
instrumented E2E. Allocation plus cleanup is 0.842 ms, and CPU Top-K is 0.275
ms. At `N=100,000, D=768`, the kernel is 64.144 ms, or 58.3%, and database
H2D is 31.888 ms, or 29.0%. These measurements explain why kernel-only and
E2E latency tell different stories without claiming a memory- or compute-bound
cause.

## Nsight Systems Findings

Availability and capture command:

```text
Nsight Systems 2025.1.3.140-251335620677v0
scripts/profile_nsys.sh build-cuda/vector_search /tmp/vector-search-m2-nsys
```

The script captured `N=10,000,D=128` and `N=100,000,D=768` with the CUDA API
trace. With two warm-ups and one measured iteration, the API trace showed,
per report, three backend searches and therefore nine `cudaMalloc` calls, nine
`cudaMemcpy` calls, three kernel launches, and nine `cudaFree` calls. This
confirms the repeated allocation/copy/launch/free pipeline and its serialized
ordering. The trace also showed the expected event record/synchronization
calls; there are no streams or overlapping transfers in M1.

The installed Nsight Systems report did not contain device-side CUDA kernel or
GPU memory activity. Its summaries explicitly reported:

```text
SKIPPED: ... does not contain CUDA kernel data.
SKIPPED: ... does not contain GPU memory data.
```

An additional attempt to request the CUDA hardware trace returned:

```text
Illegal --trace argument 'cuda-hw'
One or more GPUs cannot be configured to collect a CUDA hardware trace.
```

Therefore the reports support API-level sequencing and call-count findings,
but not GPU busy/idle intervals, device memory throughput, or kernel timeline
duration. CUDA Event and host stage measurements are used for the quantitative
latency tables instead of treating the incomplete Nsight Systems GPU tables as
evidence.

## Nsight Compute Findings

`which ncu` does not find the command because it is not on `PATH`, but the
existing installation is available at
`/opt/nvidia/nsight-compute/2025.2.1/ncu` (Nsight Compute 2025.2.1.0). The
focused wrapper uses that absolute path when needed.

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

Nsight Compute is installed, but WSL2 GPU performance-counter access is not
permitted. No system package was installed and no security control was
bypassed. NVIDIA GPU Performance Counter access may need to be enabled on the
Windows host; this task does not change Windows NVIDIA Control Panel settings.
The following M2 metrics therefore remain unavailable: achieved occupancy, SM
utilization, global-load efficiency, device memory throughput, compute
throughput, instruction/warp state, and profiler-reported register usage.

As a separate compiler diagnostic (not an Nsight Compute measurement), an
`sm_86` `nvcc -Xptxas=-v` compile of the unchanged source reported:

```text
Used 22 registers, used 0 barriers, 0 bytes stack frame,
0 bytes spill stores, 0 bytes spill loads
```

The source/runtime launch configuration is fixed at 256 threads per block.
For `Q=4`, the observed grid is `ceil(Q*N/256)`: 157 blocks at `N=10,000`
and 1,563 blocks at `N=100,000`. These are launch facts, not evidence that
the kernel is memory-bound or compute-bound.

## Crossover Analysis

The normal, uninstrumented benchmark sweep is in
`benchmarks/results/m2_crossover.csv`. Speedup is CPU E2E divided by CUDA E2E;
values above 1 mean CUDA E2E is faster.

| N | CPU E2E | CUDA kernel | CUDA E2E | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1.735 ms | 0.198 ms | 2.030 ms | 0.855x |
| 15,000 | 2.539 ms | 0.413 ms | 2.462 ms | 1.031x |
| 20,000 | 3.377 ms | 0.607 ms | 3.317 ms | 1.018x |
| 40,000 | 7.384 ms | 1.288 ms | 6.278 ms | 1.176x |
| 60,000 | 12.067 ms | 1.998 ms | 8.753 ms | 1.379x |
| 80,000 | 15.394 ms | 2.731 ms | 11.016 ms | 1.397x |
| 100,000 | 20.687 ms | 3.526 ms | 13.944 ms | 1.484x |

CPU is faster at 10K and CUDA is faster at 15K and above in this selected
run, so the empirical crossover lies between 10K and 15K vectors. Linear
interpolation of the signed E2E difference estimates approximately 14K, but
the 15K margin is small and should be treated as noise-sensitive rather than
as a universal threshold.

The stage data provides the explanation supported by measurement: at 10K the
kernel is sub-millisecond while allocation, transfers, CPU Top-K, cleanup, and
other E2E work remain about two milliseconds. At 100K×128 the kernel grows to
3.523 ms and the database H2D plus CPU Top-K stages grow to 7.015 ms, so the
fixed/per-search work is amortized. Nsight Systems also confirms that these
stages are repeated for every search. The data does not isolate one single
launch-overhead number because M1 does not time kernel launch as a separate
stage.

## Dimension Analysis

The normal dimension sweep is in `benchmarks/results/m2_dimension.csv` and
uses `N=100,000`, `Q=4`, `K=10`, seed `42` without detailed timing
instrumentation:

| D | CPU E2E | CUDA kernel | CUDA E2E | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 21.952 ms | 3.189 ms | 13.784 ms | 1.593x |
| 256 | 41.090 ms | 18.842 ms | 36.366 ms | 1.130x |
| 768 | 141.561 ms | 64.300 ms | 107.664 ms | 1.315x |

The corresponding instrumented decomposition is:

| D | Allocation | H2D DB | H2D Query | Kernel | D2H Scores | CPU Top-K | Cleanup | Instrumented E2E |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.837 | 4.534 | 0.129 | 3.523 | 0.287 | 2.481 | 0.656 | 13.885 ms |
| 256 | 1.320 | 10.860 | 0.130 | 18.909 | 0.486 | 2.202 | 1.057 | 37.760 ms |
| 768 | 1.694 | 31.888 | 0.116 | 64.144 | 0.612 | 2.398 | 2.149 | 110.006 ms |

Observed scaling is not proportional to dimension: from D=128 to D=256,
kernel latency rises about 5.9x for a 2x dimension increase; from D=256 to
D=768 it rises about 3.4x for a 3x increase. Database H2D rises from 4.534 to
10.860 to 31.888 ms, consistent with the increasing row-major byte volume,
while the D2H score array is the same size for all three rows because N and Q
are fixed. The D=256 behavior is a repeatable observation in this run, not a
causal explanation.

The timing evidence shows that both data movement and dot-product work become
material as D grows. Nsight Compute was unavailable, so the measurements do
not determine whether the kernel scaling is caused by global-memory
coalescing, cache behavior, instruction throughput, occupancy, or another
device effect.

## Confirmed Bottlenecks

Ranked by measured impact in the selected workloads:

1. The high-dimensional similarity kernel is the largest measured stage at
   `N=100,000,D=768` (64.144 ms, 58.3% of instrumented E2E). This is a
   CUDA-event/timing conclusion, not a memory-bound or compute-bound label.
2. H2D database transfer is the largest non-kernel stage at the same workload
   (31.888 ms, 29.0%); it is also 4.534 ms (32.7%) at D=128. This is measured
   host-to-device transfer latency, not device global-memory throughput.
3. Small-workload per-search overhead is material: at 10K×128, allocation and
   cleanup total 0.842 ms, CPU Top-K is 0.275 ms, and the kernel is only
   0.195 ms. The API timeline confirms that allocation, copies, launch, and
   cleanup are repeated for each search.

## Rejected Hypotheses

* A full D2H score copy is not the dominant large-workload bottleneck in these
  measurements: it is 0.612 ms at 100K×768, about 0.6% of instrumented E2E.
* The claim that the CUDA kernel alone explains the small-workload crossover
  is rejected by the 10K stage breakdown; the kernel is a small fraction of
  E2E while non-kernel work is comparable to the CPU result.
* The row-major/adjacent-thread strided-access hypothesis is not confirmed or
  disproved. It requires Nsight Compute global-load efficiency and memory
  metrics that were unavailable, so no mapping redesign is justified as an
  M2 conclusion.

## M3 Candidates

The single highest-priority next experiment is a separate backend that tests a
global-memory-friendly thread/data mapping for the similarity kernel, after
Nsight Compute metrics are available. The reason is quantitative: the frozen
kernel accounts for 58.3% of the high-dimensional E2E and exhibits the largest
absolute stage, while the current data does not yet identify whether its cause
is access efficiency or instruction/occupancy behavior. The M1 backend must
remain unchanged as the comparison point.

Secondary candidates, in descending measured opportunity, are:

* keeping the database device-resident or otherwise reducing the 29.0% large
  database H2D stage;
* reducing CPU Top-K and score movement after the kernel (especially the
  17.9% CPU Top-K stage at 100K×128); and
* reusing device allocations to reduce the small-workload allocation/cleanup
  contribution.

These are recommendations only. No M3 kernel, transfer, allocation, stream,
layout, or Top-K optimization was implemented in M2.

# M2.5 — Nsight Compute Analysis

M2.5 completes the profiler evidence that was unavailable during M2. This is
an analysis-only milestone: the M1 `cuda-naive` kernel, row-major database
layout, one-thread-per-query/vector mapping, sequential dimension loop, fixed
256-thread block, full score copy, and CPU Top-K are unchanged.

## Profiling Environment

The repository was verified inside WSL2 at
`/home/sungj/projects/CUDA-Accelerated-Vector-Search-Engine`, not under
`/mnt/d`.

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB, compute capability 8.6 |
| Driver | 576.88 |
| CUDA Toolkit / NVCC | 12.9 / 12.9.86 |
| GCC / CMake | GCC 11.4.0 / CMake 3.22.1 |
| Nsight Compute | 2025.2.1.0 |
| Primary workload | `N=100,000, D=768, Q=4, K=10, seed=42` |
| Comparison workload | `N=100,000, D=128, Q=4, K=10, seed=42` |

`nvcc` and `ncu` were invoked from their WSL installations because they were
not on the default PATH:
`/usr/local/cuda-12.9/bin` and
`/opt/nvidia/nsight-compute/2025.2.1`.

The focused permission probe completed successfully and produced
`profiling/ncu/m2p5_permission_probe.ncu-rep`; it did not report
`ERR_NVGPUCTRPERM`. The reproducible wrapper is
`scripts/profile_ncu.sh`. It collects `SpeedOfLight`,
`MemoryWorkloadAnalysis`, `MemoryWorkloadAnalysis_Tables`, `Occupancy`,
`LaunchStats`, `SchedulerStats`, `InstructionStats`, and
`WarpStateStats`, then emits a concise CSV summary.

## Validation

Release CPU and CUDA configurations were reconfigured and built for
`CMAKE_CUDA_ARCHITECTURES=86`. Both test suites passed:

* CPU CTest: 1/1 passed.
* CUDA CTest: 2/2 passed, including the CUDA correctness test.
* A normal `cuda-naive` search at `N=10,000,D=128,Q=4,K=10,seed=42` completed.
  The CPU and CUDA result checksums were `117359.130` and `117359.128`,
  respectively, within the existing FP32 tolerance.

The normal unprofiled validation runs used two warm-ups and five measured
iterations. They reported `60.130 ms` kernel / `106.778 ms` E2E at D=768 and
`3.493 ms` kernel / `14.080 ms` E2E at D=128. Nsight-instrumented application
timings are not used as benchmark results; the profiler's kernel `Duration`
metric is used only as profiler evidence.

## Launch Configuration

For the primary workload, `Q*N=400,000` pairs, so the unchanged launch is
`grid=(1,563,1)`, `block=(256,1,1)`, and `400,128` launched threads. Nsight
Compute reported:

| Metric | D=768 primary profile |
| --- | ---: |
| Registers per thread | 28 |
| Static shared memory per block | 0 bytes |
| Dynamic shared memory per block | 0 bytes |
| Driver-reserved shared memory per block | 1.02 KiB |
| Theoretical occupancy | 100.0% |
| Achieved occupancy | 95.43% |
| Theoretical active warps/SM | 48 |
| Achieved active warps/SM | 45.81 |
| Occupancy block limits | SM 16, registers 8, shared memory 8, warps 6 |

The direct `nvcc -Xptxas=-v` diagnostic for the unchanged `sm_86` source
reported 22 registers, zero barriers, and zero spills. Nsight Compute's
runtime launch statistic reported 28 registers/thread; the latter is the value
used for the profiled launch. The discrepancy is retained as a build/link
diagnostic rather than silently treating the two tools as identical measures.

Occupancy is therefore healthy and not the primary limitation: the launch is
theoretically full and achieves about 95%. The warps-per-block resource, not
register or shared-memory usage, is the limiting theoretical block count.

## Speed-of-Light and Compute Analysis

The final wrapper-run primary report measured:

| Metric | D=768 | D=128 comparison |
| --- | ---: | ---: |
| Compute (SM) throughput | 0.26% of peak | 6.39% |
| Memory / DRAM throughput | 27.19% of peak | 58.14% |
| DRAM rate | 47.81 GB/s | 102.24 GB/s |
| Memory busy | 13.38% | 52.35% |
| L1/TEX hit rate | 73.91% | 76.24% |
| L2 hit rate | 11.33% | 13.15% |
| Nsight Compute kernel duration | 55.42 ms | 3.46 ms |

The exact cycle-normalized utilization values vary across Nsight Compute
replay passes, but every focused pass remained far below compute peak and
below saturated DRAM bandwidth for D=768. Nsight Compute's Speed-of-Light
guidance classifies simultaneous low compute and bandwidth utilization as a
latency issue. The evidence does not support calling this a pure compute-
throughput or pure DRAM-bandwidth-bound kernel.

## Memory Analysis

The decisive global-load metric is
`Average Bytes Per Sector For Global Loads = 4.0 bytes/sector`, with a
32-byte maximum sector. Nsight Compute's memory rule states that only 4 of
the 32 transmitted bytes are utilized per thread and explicitly flags a
possible stride between threads. The same 4/32 result appeared in the D=128
comparison. The primary report also noted that 87.3% of the relevant sectors
missed in L2.

For the unchanged row-major mapping, adjacent threads at a fixed loop
iteration access:

```text
thread 0: database[0*768 + d]
thread 1: database[1*768 + d]
...
```

so adjacent addresses are separated by `768*4 = 3,072` bytes. A warp's
database loads therefore touch separate 32-byte sectors while using only one
4-byte word in each sector, matching the measured 4/32 sector utilization.
The query loads are reused across threads and can hit in cache; that does not
remove the database-load transaction inefficiency.

The L1/TEX hit rate near 74% is not evidence of coalescing: cache reuse can
hide some requests, while the sector-utilization metric directly measures
that the transmitted sectors are poorly used. The D=128 result shows that the
mapping problem is present even when the loop is short.

## Warp Analysis

For D=768, Nsight Compute reported approximately 11.11 active warps per
scheduler but only 0.05 eligible warps per scheduler. Only 2.01% of scheduler
cycles had one or more eligible warps; 97.99% had no eligible warp, and the
issued-warp rate was 0.02 per scheduler. Warp cycles per issued instruction
were 551.48.

The dominant aggregate stall categories were:

* LG-memory instruction-queue throttle: 333.4 cycles, about 60.46%.
* L1TEX scoreboard dependency: 219.0 cycles, about 39.71%.

The D=128 comparison had the same qualitative pattern: 96.58% no-eligible
cycles, 0.07 eligible warps per scheduler, 301.21 warp cycles per issued
instruction, L1TEX scoreboard stalls about 52.3%, and LG-memory queue stalls
about 45.4%. It retained 28 registers/thread and 100% theoretical occupancy,
with 85.67% achieved occupancy and 41.12 active warps/SM. The optional
source-level warp-sampling metric was unavailable, so the report does not claim a
source-line stall attribution. Small residual stall categories are not
interpreted.

## Memory-Access Hypothesis

**Classification: confirmed.**

The 4/32 average bytes-per-sector result, the explicit Nsight Compute
stride warning, the low L2 hit rate, and the dominant LG/L1TEX stall reasons
all support inefficient global-load transactions from the current
thread/data mapping. This confirms the access inefficiency hypothesis; it
does not mean that DRAM bandwidth is saturated.

## Sequential-D Hypothesis

**Classification: partially supported.**

The D=768 profile executed `77,612,532` instructions versus `13,612,532` at
D=128, and warp cycles per issued instruction rose from about `301` to `551`.
The long per-thread loop therefore creates substantially more repeated
load/accumulate work and exposes more memory-latency time. However, compute
throughput stayed in the low single digits and the dominant stall categories
were memory queue/scoreboard stalls rather than a measured execution-
dependency category. Nsight Compute does not prove that the scalar
accumulator dependency alone causes the kernel cost.

Dimension-parallel execution is consequently a justified M3 experiment, but
the profile supports it as a combined memory-mapping and latency-hiding
experiment rather than as a proven standalone fix.

## M3 Decision

### Primary experiment: Candidate A — block-per-vector plus parallel dimension reduction

Implement a separate experimental backend in which one block handles one
query/database-vector pair, threads load adjacent dimensions, and a
block-level reduction produces the score. This is the single most justified
kernel experiment because it directly addresses both strongest signals:

1. the current database loads use only 4/32 bytes per sector because adjacent
   threads walk different rows; and
2. the long D loop leaves almost all scheduler issue slots without an eligible
   warp and increases warp latency from about 301 to 551 cycles.

Expected mechanism: adjacent threads should request adjacent dimensions, and
the D work can be distributed across the block rather than executed serially
by one thread. Tradeoffs are synchronization and reduction instructions,
more blocks (`Q*N` blocks), changed register/shared-memory use, and possible
underutilization or overhead for short dimensions.

The validation benchmark should keep `N=100,000,D=768,Q=4,K=10,seed=42`,
Release build, two warm-ups, and five measured iterations, and should compare
CPU correctness, CUDA-event kernel latency, and full E2E latency against the
frozen M1 `cuda-naive` backend. The experiment must report memory-sector
utilization, achieved occupancy, warp stalls, kernel latency, and E2E latency.

### Secondary conceptual candidates

* **Candidate B — database layout transformation:** a dimension-major
  database could make adjacent threads access adjacent elements while
  retaining one thread per vector. It directly targets the confirmed
  transaction inefficiency, but adds a transpose/preprocessing cost and
  changes the H2D representation; it does not parallelize the D loop.
* **Candidate C — multiple vectors per block or warp-oriented mapping:** a
  warp/subgroup per vector could coalesce dimension loads and reduce within a
  warp, with the mapping naturally spanning D=128/256/768. It may reduce
  block-level overhead but introduces subgroup-tail and reduction design
  choices. It remains a later comparison, not the selected primary.

Persistent database residency is a separate system-level optimization because
the M2 stage table measured about 29% of E2E in H2D database transfer at
100K×768. It must not be combined with the primary kernel experiment.

No M3 candidate was implemented in M2.5.

# M3 — Block-per-Vector Parallel Reduction

## Hypothesis

M2.5 confirmed that the frozen cuda-naive kernel has poor global-load
sector utilization and severe memory-latency exposure. Adjacent threads in
M1 access different database rows, so the D=768 stride between neighboring
threads is 3,072 bytes and the profiler measured only 4 useful bytes per
32-byte sector for global loads. M2.5 also measured dominant LG-memory queue
and L1TEX scoreboard stalls, with very few eligible warps.

M3 tests whether assigning one block to one query/database-vector pair, then
having threads cooperatively process adjacent dimensions, improves
coalescing and makes more dimension work available to the scheduler.

## Design

The new cuda-block backend uses:

~~~text
one CUDA block -> one (query_idx, vector_idx) pair
thread t       -> dimensions t, t + 256, t + 512, ...
partial sums   -> 256-element shared-memory array
reduction      -> synchronized tree reduction
~~~

The grid contains exactly Q*N blocks. Every thread writes one partial sum
before the first barrier, so dimensions smaller than the block size are
valid. The reduction uses explicit __syncthreads() barriers and no CUB,
Thrust, cuBLAS, warp shuffles, or other reduction library. The block size is
fixed at 256 threads; it was not tuned in this milestone.

The M1 src/cuda/naive_search.cu source and cuda-naive factory behavior
were left unchanged. cuda-block is a separate backend selected through the
same CLI and SearchBackend interface.

## Controlled Variables

The following remain unchanged between M1 and M3:

* per-search device allocation and cleanup;
* synchronous H2D database copy;
* synchronous H2D query copy;
* full score-array D2H copy;
* CPU Top-K selection;
* CUDA Event kernel timing boundaries;
* fixed 256 threads/block; and
* normalized deterministic FP32 input generation.

No persistent database, allocation reuse, streams, pinned memory,
cudaMemcpyAsync, GPU Top-K, query caching, layout transformation,
vectorized loads, or multiple-vector block mapping was added.

## Correctness

The CPU backend remains the oracle. The CUDA Release test suite passed 2/2
tests, and the CPU Release suite passed 1/1. The M3 CUDA cases compare both
CUDA backends against CPU with tolerance-based FP32 checks and cover:

* known dot products with Q>1 and K>1;
* K=1;
* D=3, 127, 128, 255, 256, 257, and 768;
* dimensions smaller than, equal to, larger than, and not divisible by 256;
* multiple queries; and
* stage timing enabled for cuda-block.

The different accumulation order is therefore accepted within tolerance;
bitwise equality is not required.

## Performance

The benchmark runner is
[scripts/run_m3_benchmarks.py](../scripts/run_m3_benchmarks.py). It uses
Release binaries, seed 42, Q=4, K=10, two warm-ups, and five measured
iterations. Dataset generation is excluded. The table below contains
side-by-side values from the same benchmark environment; speedups are
naive / block, so values above 1 are improvements.

| Workload | CPU E2E (ms) | Naive kernel (ms) | Block kernel (ms) | Kernel speedup | Naive E2E (ms) | Block E2E (ms) | E2E speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary N=100,000, D=768 | 162.050 | 63.835 | 7.993 | 7.986x | 111.474 | 56.139 | 1.986x |
| N=100,000, D=128 | 24.470 | 3.544 | 5.476 | 0.647x | 13.969 | 16.459 | 0.849x |
| N=100,000, D=256 | 48.888 | 18.923 | 5.637 | 3.357x | 37.082 | 23.640 | 1.569x |
| N=100,000, D=768 | 151.464 | 63.524 | 7.994 | 7.946x | 111.358 | 52.090 | 2.138x |
| Small N=10,000, D=128 | 1.866 | 0.184 | 0.539 | 0.341x | 1.864 | 2.549 | 0.731x |

The primary run therefore improves kernel latency by about 8.0x and E2E
latency by about 2.0x. The D=128 and 10K workloads are slower because the
fixed 256-thread block still pays for shared-memory reduction and barriers
when each pair has little dimension work. This is a measured tradeoff, not a
reason to tune M3 beyond the planned experiment.

Machine-readable records are
[m3_primary.csv](../benchmarks/results/m3_primary.csv),
[m3_dimension.csv](../benchmarks/results/m3_dimension.csv),
[m3_small_workload.csv](../benchmarks/results/m3_small_workload.csv), and
the combined
[m3_comparison.csv](../benchmarks/results/m3_comparison.csv).

## Profiler Comparison

Both focused profiles used the M2.5 sections
SpeedOfLight, MemoryWorkloadAnalysis,
MemoryWorkloadAnalysis_Tables, Occupancy, LaunchStats,
SchedulerStats, InstructionStats, and WarpStateStats for
N=100,000, D=768, Q=4, K=10, seed 42, two warm-ups, and one profiled launch.
The direct metric CSVs additionally collect bytes/sector and the two
scheduler-stall counters.

| Metric | cuda-naive | cuda-block | Interpretation |
| --- | ---: | ---: | --- |
| Global-load useful bytes/32-byte sector | 4 (12.5%) | 32 (100%) | Targeted load coalescing improved 8x |
| Global-store useful bytes/32-byte sector | 32 (100%) | 4 (12.5%) | Score-store pattern is the new residual inefficiency |
| DRAM throughput | 51.41 GB/s (28.23%) | 125.76 GB/s (70.21%) | More useful load traffic reached memory |
| L1/TEX hit rate | 71.75% | 49.91% | Lower after changing the access pattern |
| L2 hit rate | 14.17% | 2.27% | Lower; M3 is not an L2-locality optimization |
| Speed-of-Light SM throughput | 2.55% | 74.74% | More reduction/arithmetic work is issued |
| Speed-of-Light memory throughput | 28.23% | 74.74% | Both memory and issue activity increased |
| Achieved occupancy | 95.46% | 93.93% | Essentially preserved; theoretical occupancy stayed 100% |
| Active warps/SM | 45.82 | 45.09 | Essentially preserved |
| Registers/thread | 28 | 24 | Lower in the block kernel |
| Static shared memory/block | 0 B | 1.02 KiB | Reduction state only |
| Executed instructions | 77,618,984 | 632,800,000 | Higher from parallel reduction/barriers |
| Eligible warps/scheduler | 0.03 | 1.31 | Strong improvement |
| No-eligible-warp cycles | 98.67% | 41.01% | Strong improvement |
| LG-throttle direct stall share | 63.47% | 0.00% | Queue-throttle stall removed in this capture |
| L1TEX long-scoreboard direct stall share | 35.71% | 38.94% | Similar share, but absolute rule cycles fell |
| L1TEX scoreboard rule stall | 310.7 cycles | 7.4 cycles | Much less absolute memory wait |
| Warp cycles/issued instruction | 860.95 | 19.02 | Large latency reduction |
| Nsight Compute kernel duration | 54.01 ms | 10.12 ms | Profiler replay duration improved |

The M3 focused report still flags 4/32 bytes per sector for global stores,
which is expected from one thread per block writing one score. The direct
global-load counter is the relevant evidence for the M2.5 hypothesis: query
and database loads now use adjacent dimensions within each warp. Nsight
Compute profile durations are reported as profiler evidence; the normal
CUDA Event benchmark table above is the performance result.

## Result

**successful**

M3 successfully validated the profiler hypothesis for the primary
high-dimensional workload: global-load utilization improved from 4 to 32
useful bytes per sector, eligible warps increased, LG-throttle and absolute
L1TEX wait decreased, kernel latency improved about 8.0x, and E2E latency
improved about 2.0x. It is not universally faster: D=128 and the small
workload regress, and L2 hit rate and score-store sector utilization worsen.
The backend is kept as a separate experimental stage so those tradeoffs
remain visible.

## Interpretation

The evidence supports two mechanisms at once. First, the adjacent-dimension
mapping fixes the confirmed row-stride load problem. Second, distributing D
across 256 threads removes the long serial per-thread loop and exposes much
more independent work. The reduction adds shared-memory traffic and barriers,
but occupancy remains healthy and the high-D workloads amortize that cost.
The low-D regressions show that M3 is a workload-shape optimization rather
than a universally dominant backend.

## Next Recommendation

The single primary M4 experiment should be a warp-per-vector reduction:
retain the coalesced adjacent-dimension loads while replacing the full
block-level reduction/barrier cost with a warp/subgroup mapping. M3's D=128
and 10K regressions, together with preserved occupancy and the remaining
score-store warning, justify testing whether a smaller reduction scope can
keep the load benefit while reducing synchronization and underutilization.
Do not combine that experiment with layout changes, persistent residency,
streams, or block-size tuning.

# M4 — Warp-per-Vector Reduction

## Motivation

M3 fixed the original global-load coalescing problem, but its fixed
256-thread block and full shared-memory tree reduction regressed at D=128 and
on the 10K workload. M4 isolates the reduction scope as the next kernel
experiment while preserving the M3 adjacent-dimension load mapping.

## Hypothesis

A warp-per-vector mapping can retain coalesced database loads because lanes
0..31 access adjacent dimensions, while reducing shared-memory traffic,
block-wide barriers, and full-block reduction work. The experiment does not
assume that the smaller reduction scope will win for every dimension.

## Design

The new cuda-warp backend is a separate implementation:

~~~text
256 threads/block = 8 warps/block
one warp -> one (query, database-vector) pair
lane l -> dimensions l, l + 32, l + 64, ...
warp reduction -> __shfl_down_sync offsets 16, 8, 4, 2, 1
~~~

The global warp ID is blockIdx.x * 8 + warp_id_in_block, and the grid is
ceil(Q*N/8). A ballot before the tail check derives a valid mask; all lanes
in a valid pair warp participate in every shuffle with that same mask. Lane 0
writes the score. The reduction uses no explicit shared memory and no
__syncthreads().

cuda-naive and cuda-block remain separate backends and were not
algorithmically changed.

## Controlled Variables

M4 keeps the following unchanged:

* per-search device allocation and cleanup;
* synchronous H2D database copy;
* synchronous H2D query copy;
* full score-array D2H copy;
* CPU Top-K selection;
* CUDA Event kernel timing boundaries;
* normalized deterministic FP32 data;
* Q=4, K=10, seed 42 for the benchmark; and
* Release build, two warm-ups, and five measured iterations.

No persistent database, allocation reuse, pinned memory, streams,
cudaMemcpyAsync, GPU Top-K, vectorized loads, layout transformation,
cuBLAS, CUB, Thrust, or block-size tuning was added.

## Correctness

The fresh CUDA Release build passed both CTest targets (2/2), and the CPU
Release build passed its CTest target (1/1). The CUDA correctness cases
compare CPU, cuda-naive, cuda-block, and cuda-warp with tolerance-based
FP32 checks for:

* D = 3, 31, 32, 33, 127, 128, 255, 256, 257, and 768;
* Q=1 and Q>1;
* K=1 and K>1;
* N=11 and Q=2, giving 22 pairs and a partially used eight-warp block;
* N=257 and Q=1, giving a non-multiple-of-eight pair count; and
* stage timing enabled for cuda-warp.

## Performance

The M4 runner is
[scripts/run_m4_benchmarks.py](../scripts/run_m4_benchmarks.py). It recorded
normal CUDA Event kernel timing and backend end-to-end timing separately.
Values below are the measured averages from the fresh build.

| Workload | Naive kernel | Block kernel | Warp kernel | Warp/Naive kernel | Warp/Block kernel | Naive E2E | Block E2E | Warp E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N=100,000, D=128 | 3.388 ms | 5.712 ms | 1.123 ms | 3.017x | 5.086x | 14.291 ms | 16.460 ms | 11.562 ms |
| N=100,000, D=256 | 19.149 ms | 5.559 ms | 2.351 ms | 8.145x | 2.365x | 37.141 ms | 23.180 ms | 19.898 ms |
| N=100,000, D=768 | 64.447 ms | 7.879 ms | 6.918 ms | 9.316x | 1.139x | 108.782 ms | 51.515 ms | 50.381 ms |
| N=10,000, D=128 | 0.202 ms | 0.530 ms | 0.123 ms | 1.642x | 4.309x | 2.345 ms | 2.547 ms | 2.050 ms |

For the primary D=768 workload, warp improved kernel latency 9.316x versus
naive and 1.139x versus block. Its E2E latency was 2.159x lower than naive
and 1.023x lower than block. At D=128, warp removed the M3 kernel regression
and improved E2E by 1.424x versus block. On the 10K D=128 check, warp
improved E2E by 1.242x versus block, although allocation and transfer
overhead still dominate the absolute latency.

The complete records are
[m4_primary.csv](../benchmarks/results/m4_primary.csv),
[m4_dimension.csv](../benchmarks/results/m4_dimension.csv),
[m4_small_workload.csv](../benchmarks/results/m4_small_workload.csv), and
[m4_comparison.csv](../benchmarks/results/m4_comparison.csv).

## Dimension Sweep

The additional N=100,000, Q=4, K=10 sweep used D=32, 64, 128, 256, 512,
768, and 1024. Warp was the fastest kernel at D=64 through D=1024. Naive
was fastest at D=32. Warp was also the fastest E2E backend at D=64, 128,
256, 512, and 1024; naive was fastest at D=32, and block was marginally
faster than warp at D=768 E2E (51.193 ms versus 51.987 ms).

| D | Naive kernel | Block kernel | Warp kernel | Best kernel | Naive E2E | Block E2E | Warp E2E | Best E2E |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 32 | 0.521 ms | 5.347 ms | 0.888 ms | naive | 6.511 ms | 11.129 ms | 6.902 ms | naive |
| 64 | 1.378 ms | 6.132 ms | 0.726 ms | warp | 8.677 ms | 13.864 ms | 7.938 ms | warp |
| 128 | 3.388 ms | 5.712 ms | 1.123 ms | warp | 14.291 ms | 16.460 ms | 11.562 ms | warp |
| 256 | 19.149 ms | 5.559 ms | 2.351 ms | warp | 37.141 ms | 23.180 ms | 19.898 ms | warp |
| 512 | 43.011 ms | 6.815 ms | 4.550 ms | warp | 73.915 ms | 37.670 ms | 35.529 ms | warp |
| 768 | 63.601 ms | 7.975 ms | 6.985 ms | warp | 109.778 ms | 51.193 ms | 51.987 ms | block |
| 1024 | 88.739 ms | 10.016 ms | 9.268 ms | warp | 146.875 ms | 68.729 ms | 66.828 ms | warp |

## Nsight Compute

The focused wrapper reused the M2.5/M3 sections
SpeedOfLight, MemoryWorkloadAnalysis, MemoryWorkloadAnalysis_Tables,
Occupancy, LaunchStats, SchedulerStats, InstructionStats, and WarpStateStats.
Both backends were profiled with N=100,000, Q=4, K=10, seed 42 at D=768
and D=128. The reports are under
[profiling/ncu/m4](../profiling/ncu/m4/).

### D=768

| Metric | cuda-block | cuda-warp |
| --- | ---: | ---: |
| Global-load useful bytes/32-byte sector | 32 (100%) | 32 (100%) |
| Global-store useful bytes/32-byte sector | 4 (12.5%) | 4 (12.5%) |
| DRAM memory throughput | 125.00 Gbyte/s (70.22%) | 177.48 Gbyte/s (97.09%) |
| L1/TEX hit rate | 49.91% | 50.18% |
| L2 hit rate | 2.24% | 2.23% |
| Achieved occupancy | 93.95% | 89.31% |
| Achieved active warps/SM | 45.10 | 42.87 |
| Registers/thread | 24 | 38 |
| Static shared memory/block | 1.02 KiB | 0 B |
| Eligible warps/scheduler | 1.31 | 0.39 |
| No-eligible cycles | 41.00% | 73.72% |
| Warp cycles/issued instruction | 19.01 | 40.98 |
| Executed instructions | 632,800,000 | 202,403,225 |
| Dominant direct stall | L1TEX scoreboard, 7.4 cycles (38.90%) | L1TEX scoreboard, 36.7 cycles (89.53%) |
| Nsight Compute replay duration | 10.17 ms | 7.06 ms |

The warp kernel preserved M3's 32/32 load-sector result and removed the
1.02 KiB static reduction array. It executed fewer instructions, but used
more registers and had lower achieved occupancy. At D=768, the profiler
shows worse scheduler eligibility and more L1TEX scoreboard stall for warp;
the normal five-iteration CUDA-event result nevertheless remained
competitive and was 1.139x faster at kernel scope.

### D=128

| Metric | cuda-block | cuda-warp |
| --- | ---: | ---: |
| Global-load useful bytes/32-byte sector | 32 (100%) | 32 (100%) |
| DRAM memory throughput | 31.16 Gbyte/s (17.06%) | 185.12 Gbyte/s (96.60%) |
| L1/TEX hit rate | 49.61% | 50.86% |
| L2 hit rate | 12.00% | 5.24% |
| Achieved occupancy | 91.32% | 86.24% |
| Achieved active warps/SM | 43.83 | 41.39 |
| Registers/thread | 24 | 38 |
| Static shared memory/block | 1.02 KiB | 0 B |
| Eligible warps/scheduler | 1.18 | 1.35 |
| No-eligible cycles | 44.07% | 38.85% |
| Warp cycles/issued instruction | 19.45 | 15.91 |
| Executed instructions | 444,000,000 | 80,400,000 |
| Dominant direct stall | CTA barrier, 6.6 cycles (33.9%) | L1TEX scoreboard, 8.3 cycles (52.46%) |

At D=128, the warp profile supports the intended reduction-overhead result:
barrier stalls disappear, executed instructions fall, eligible warps rise,
and no-eligible cycles fall. L2 hit rate is lower, so the result is not an
L2-locality improvement.

## Result

**successful**

M4 answered the four experimental questions positively for the primary
purpose, with workload-dependent limits:

1. Coalescing was preserved: both M3 and M4 measured 32 useful load bytes per
   32-byte global-load sector.
2. Small-D performance improved: at D=128, warp was 5.086x faster than block
   in kernel time and 1.424x faster E2E; at 10K x D=128 it was 4.309x faster
   in kernel time and 1.242x faster E2E.
3. High-D performance remained competitive: at D=768, warp was 1.139x faster
   than block in kernel time and 1.023x faster E2E on the primary run.
4. Reduction scope was reduced in the implementation and profiler evidence:
   static shared reduction state fell from 1.02 KiB to zero, and D=128
   barrier/instruction overhead fell materially.

M4 is not a universal replacement: D=32 still favored naive, D=768 E2E was
within measurement noise and favored block in the dimension-sweep row, and
the D=768 profile showed higher register use, lower scheduler eligibility,
and larger L1TEX scoreboard stalls for warp. These are measured tradeoffs,
not reasons to alter the controlled first experiment.

## Interpretation

The warp mapping retained the main M3 memory-access benefit while making the
reduction local to the 32 lanes that own one pair. At smaller D, this avoided
making 224 lanes participate in a full block reduction and removed the
barrier-heavy instruction cost that caused the M3 regression. At D=768, the
warp kernel still won the normal kernel benchmark, but the profiler shows
that fewer independent pair blocks and higher register pressure can reduce
latency hiding. The result therefore supports warp-per-vector as a useful
experimental backend, not as an automatic backend-selection rule.

## M5 Recommendation

Run exactly one next experiment: a controlled warp-per-vector
warps-per-block/block-size sweep to test whether the D=768 scheduler
regression (73.72% no-eligible cycles versus 41.00% for cuda-block) is
caused by the fixed eight-warps-per-block launch configuration. Keep the
warp mapping, transfers, allocation lifecycle, CPU Top-K, and benchmark
protocol unchanged. Do not implement that experiment in M4.


# M5 — GPU-Resident Database

## M4 D=768 Summary Reconciliation

The M4 primary and dimension-sweep D=768 values came from different
benchmark subprocess runs. The primary record in
benchmarks/results/m4_primary.csv measured:

~~~text
cuda-block: kernel 7.879 ms, E2E 51.515 ms
cuda-warp:  kernel 6.918 ms, E2E 50.381 ms
~~~

The independent D=768 row in benchmarks/results/m4_dimension.csv measured:

~~~text
cuda-block: kernel 7.975 ms, E2E 51.193 ms
cuda-warp:  kernel 6.985 ms, E2E 51.987 ms
~~~

The dimension-sweep script invokes each workload/backend as a separate
process. The 0.794 ms D=768 E2E difference is approximately 1.5% of the
measured latency and is ordinary run-to-run variation for these host-side
and driver-sensitive timings. The dimension summary selected cuda-block
correctly for its own CSV row; the summary script did not select the wrong
backend, and neither CSV was changed to force agreement. The documentation
now labels the values as independent runs.

## Motivation

M4 reduced the warp kernel to about 6.9 ms at N=100,000, D=768, while the
end-to-end search remained about 50 ms. Earlier M2 decomposition measured
about 31.9 ms for the database H2D stage at this workload. The optimized
kernel therefore shifted attention to the request-level allocation and
database data movement path.

M5 is the first system-level optimization milestone. It does not change the
similarity algorithm or optimize the CUDA kernel further.

## Hypothesis

For multiple query batches against the same database, allocating device
storage and uploading the FP32 database once should reduce steady-state
latency by removing the repeated database allocation and database H2D copy
from each request. A one-shot search is not expected to benefit because it
must still pay database preparation.

## Architecture

The existing cuda-warp backend remains the M4 stateless baseline:

~~~text
search()
  -> allocate database/query/score buffers
  -> database H2D
  -> query H2D
  -> frozen M4 warp kernel
  -> score D2H
  -> CPU Top-K
  -> cleanup
~~~

The new cuda-warp-resident backend exposes an explicit stateful lifecycle
through ResidentSearchBackend:

~~~text
create
  -> prepare_database(DatabaseRequest)
  -> search(SearchRequest) x many
  -> reload_database(DatabaseRequest) or clear_database()
  -> destroy
~~~

prepare_database validates and uploads the database once, then keeps its
device allocation alive. Repeating prepare_database with the same host
pointer, N, and D is idempotent. reload_database explicitly reuploads,
including for the same host pointer. A different database pointer, vector
count, or dimension is a new database. search validates the prepared
dimensions and accepts either a null database pointer or the same prepared
host pointer. clear_database releases the resident allocation.

Query and score buffers are still allocated and freed per request so the
controlled variable remains database residency. The resident search timing
separates:

~~~text
Database preparation:
  db_allocation_ms
  db_h2d_ms
  prepare_total_ms

Warm query:
  query_allocation_ms
  query_h2d_ms
  kernel_ms
  query_d2h_scores_ms
  query_cpu_topk_ms
  query_cleanup_ms
  query_e2e_ms
~~~

CUDA buffers and events use RAII. All CUDA allocations, copies, event
operations, memory-info calls, kernel launches, synchronizations, and frees
are checked. A failed preparation leaves the backend unprepared; search
before preparation and incompatible requests fail explicitly.

## Controlled Experiment

The benchmark configuration was:

~~~text
N=100,000
D=768 (primary)
Q=4
K=10
seed=42
Release
warmup batches=2
measured batches=100
backend comparison: cuda-warp vs cuda-warp-resident
~~~

The dimension sweep repeated the same protocol for D=128, D=256, and D=768.
The optional workload was N=250,000, D=768. Each backend received the same
deterministic database and the same sequence of generated query batches.
Dataset generation was excluded from the timed search path.

The runner is scripts/run_m5_benchmarks.py. It writes
benchmarks/results/m5_cold_start.csv, m5_warm.csv, m5_amortization.csv,
m5_dimension.csv, and m5_comparison.csv.

## Database Preparation and Memory Footprint

The database is FP32, so bytes = N * D * 4. The resident backend also
records free device memory immediately before and after preparation. The
formal run observed 3,449,500,468 bytes free before preparation.

| Workload | Database bytes | Database MiB | DB allocation (ms) | DB H2D (ms) | Prepare total (ms) | Free before (bytes) | Free after (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary N=100,000, D=768 | 307,200,000 | 292.969 | 1.070 | 367.040 | 660.766 | 3,449,500,468 | 3,141,219,124 |
| N=100,000, D=128 | 51,200,000 | 48.828 | 0.277 | 58.707 | 362.981 | 3,449,500,468 | 3,397,071,668 |
| N=100,000, D=256 | 102,400,000 | 97.656 | 0.429 | 121.751 | 432.986 | 3,449,500,468 | 3,346,740,020 |
| N=100,000, D=768 | 307,200,000 | 292.969 | 1.794 | 368.860 | 683.011 | 3,449,500,468 | 3,141,219,124 |
| N=250,000, D=768 | 768,000,000 | 732.422 | 3.075 | 921.983 | 1,242.559 | 3,449,500,468 | 2,679,845,684 |

The optional 250K workload fit within the 4 GiB RTX 3050 Ti Laptop GPU; no
CPU fallback was used.

## Cold-Start Results

Cold resident cost is preparation plus the first query. The dedicated
one-shot runs used warmup=0 and one measured query batch. The resident
preparation measurement includes the CUDA runtime/context work observed on
that process, which is why the DB H2D event itself is much smaller than
prepare_total_ms.

| Workload | cuda-warp E2E | DB prepare | Resident first query | Resident cold total | Resident/stateless |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary N=100K, D=768 | 526.642 ms | 660.766 ms | 15.873 ms | 676.639 ms | 0.778x |
| N=100K, D=128 | 168.031 ms | 362.981 ms | 9.331 ms | 372.313 ms | 0.451x |
| N=100K, D=256 | 233.638 ms | 432.986 ms | 9.934 ms | 442.921 ms | 0.527x |
| N=100K, D=768 | 496.184 ms | 683.011 ms | 15.672 ms | 698.683 ms | 0.710x |
| N=250K, D=768 | 1,085.765 ms | 1,242.559 ms | 37.961 ms | 1,280.520 ms | 0.848x |

Values below 1.0x are cold-start regressions, not speedups. The resident
lifecycle is therefore not an optimization for one-shot requests.
The final formal run retained its raw process-level CUDA initialization and
transfer timing; its stateless D=768 database H2D was much higher than the
earlier M2 sample, so the paired within-run comparison is the meaningful result.

## Warm / Steady-State Results

The following are means over 100 measured sequential query batches after two
warmups. The resident preparation cost is excluded from the warm latency.

| Workload | cuda-warp mean | resident mean | Warm speedup | cuda-warp median | resident median |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary N=100K, D=768 | 395.504 ms | 13.289 ms | 29.761x | 395.605 ms | 13.391 ms |
| N=100K, D=128 | 72.177 ms | 6.986 ms | 10.331x | 72.042 ms | 6.966 ms |
| N=100K, D=256 | 136.776 ms | 8.278 ms | 16.523x | 136.710 ms | 8.384 ms |
| N=100K, D=768 | 394.634 ms | 13.437 ms | 29.369x | 394.637 ms | 13.341 ms |
| N=250K, D=768 | 986.258 ms | 33.958 ms | 29.043x | 985.571 ms | 33.804 ms |

For the primary M5 result, the same-run D=768 stateless mean was 395.504 ms
and resident warm mean was 13.289 ms, a 29.761x speedup. Compared with the
recorded M4 primary cuda-warp E2E of 50.381 ms, resident warm latency is
73.6% lower. In the controlled M5 run, the stateless-to-resident reduction
was 96.6%. The stateless database H2D stage averaged 369.096 ms, or 93.3%
of its 395.502 ms stage-timed E2E. This final run retained the raw transfer
timing; the earlier M2 sample was about 31.9 ms, illustrating
environment-sensitive host-to-device timing. Removing database allocation
and cleanup accounts for additional savings beyond the H2D stage alone.

## Amortization

For the primary workload, the amortized resident total is
prepare_total_ms plus the measured resident query times. The stateless total
is the sum of the same number of measured cuda-warp batches.

| Query batches | Stateless total | Stateless avg | Resident total incl. prepare | Resident avg incl. prepare | Total speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 396.543 ms | 396.543 ms | 687.126 ms | 687.126 ms | 0.577x |
| 2 | 792.198 ms | 396.099 ms | 700.819 ms | 350.410 ms | 1.130x |
| 5 | 1,978.387 ms | 395.677 ms | 742.382 ms | 148.476 ms | 2.665x |
| 10 | 3,953.803 ms | 395.380 ms | 807.001 ms | 80.700 ms | 4.899x |
| 20 | 7,911.911 ms | 395.596 ms | 938.998 ms | 46.950 ms | 8.426x |
| 100 | 39,550.352 ms | 395.504 ms | 2,001.565 ms | 20.016 ms | 19.760x |

Using mean warm latencies, the algebraic crossover is approximately 1.8
batches for the primary run; the first requested measured point showing a
benefit is 2 batches. The corresponding crossover estimates were about
5.3 batches at D=128, 3.1 at D=256, 1.7 at D=768, and 1.4 for N=250K,
D=768. The first requested points showing a benefit were 10, 5, 2, and 2
batches respectively. These estimates use the preparation sample from the
repeated run; the dedicated cold preparation sample was a separate process
measurement.

## Stage Timing

Primary D=768 per-batch means:

| Stage | Stateless cuda-warp | Resident warm |
| --- | ---: | ---: |
| Per-search device allocation | 1.570 ms (DB/query/score) | 0 (DB; query below) |
| Database H2D | 369.096 ms | 0 (preparation only) |
| Query allocation | included in 1.570 ms | 0.401 ms |
| Query H2D | 0.265 ms | 0.185 ms |
| Kernel | 7.459 ms | 7.038 ms |
| Score D2H | 2.321 ms | 2.136 ms |
| CPU Top-K | 2.415 ms | 2.452 ms |
| Cleanup | 3.069 ms | 0.433 ms |
| Stage E2E | 395.502 ms | 13.287 ms |

Resident preparation for the repeated run was 672.651 ms total; its DB
allocation and DB H2D components were 1.010 ms and 369.732 ms. Kernel
latency remained effectively unchanged, which is evidence that M5 isolated
database lifetime rather than changing the M4 arithmetic or launch mapping.

## Nsight Systems

The representative trace uses N=10,000, D=128, Q=2, K=3, three measured
batches, and zero warmups. It was captured with
scripts/profile_m5_nsys.sh. The reports are:

* profiling/nsys/m5/m5_cuda-warp_n10000_d128_q2.nsys-rep
* profiling/nsys/m5/m5_cuda-warp-resident_n10000_d128_q2.nsys-rep

The Nsight CUDA API summaries show:

| API | cuda-warp | cuda-warp-resident |
| --- | ---: | ---: |
| cudaMalloc | 9 | 7 |
| cudaMemcpy | 9 | 7 |
| cudaFree | 9 | 7 |
| cudaLaunchKernel | 3 | 3 |
| cudaMemGetInfo | 0 | 2 |

Stateless has three database allocations, three database H2D copies, three
query H2D copies, and three score D2H copies. Resident has one database
allocation/copy during preparation, then only the per-query query and score
copies. Thus the application stage output and the trace's API count both
confirm that the database H2D operation occurs once. The WSL capture did not
populate Nsight's GPU memory summary tables, but the CUDA API timeline and
application stage timings provide the required transfer-lifetime evidence.

## New Bottleneck

For the primary D=768 resident workload, the frozen similarity kernel
remains the largest steady-state component at 7.038 ms, about 53.0% of the
13.289 ms query E2E. CPU Top-K is next at 2.452 ms, about 18.5%. At D=128,
CPU Top-K is instead the largest component at 2.481 ms versus a 1.158 ms
kernel. The post-residency bottleneck is therefore workload-dependent; for
the required primary it is kernel execution, while lower-dimensional
requests expose CPU Top-K overhead.

M5 does not optimize either component.

## Result

**successful for the repeated-query objective; partially successful for
one-shot latency.**

The primary warm result improved 29.761x at the same M5 configuration, and
all tested dimensions plus the optional 250K workload improved by 10.331x to
29.761x. The cold resident path regressed to 0.451x–0.848x of stateless
one-shot E2E because preparation is paid up front. This is the expected
semantic tradeoff, so warm latency is not presented as startup latency.

## M6 Recommendation

Run exactly one next experiment: a CPU Top-K scaling experiment after
database residency, sweeping Q and K at D=128, D=256, and D=768 while keeping
the frozen M4 kernel and all transfers unchanged. Its purpose is to quantify
whether CPU Top-K becomes the next system bottleneck for lower-dimensional
resident workloads. Do not implement this experiment as part of M5.
