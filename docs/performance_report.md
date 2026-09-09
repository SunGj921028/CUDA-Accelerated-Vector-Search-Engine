# Performance Report — M2/M2.5 Profiling and M3 Kernel Redesign

## Project story

```text
M0 CPU
      |
      v
M1 naive CUDA
      |
      v
M2 stage decomposition
      |
      v
M2.5 Nsight Compute evidence
      |
      v
M3 profiler-driven kernel redesign
```

M2 and M2.5 keep the M1 `cuda-naive` algorithm as the reference: one thread
computes one query/database-vector pair, each thread loops sequentially over
D, the block size is 256, all scores are copied D2H, and CPU Top-K remains in
place. M3 adds a separate `cuda-block` backend so the profiler-driven kernel
redesign can be measured without replacing the baseline.

The machine-readable records are [the crossover results](../benchmarks/results/m2_crossover.csv),
[the dimension results](../benchmarks/results/m2_dimension.csv), and [the stage timing results](../benchmarks/results/m2_stage_timing.csv).

## Environment

| Item | Value |
| --- | --- |
| Repository | `/home/sungj/projects/CUDA-Accelerated-Vector-Search-Engine` |
| Distribution | Ubuntu 22.04.4 LTS under WSL2 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB, compute capability 8.6 |
| Driver | 576.88 |
| CUDA Toolkit | 12.9 |
| NVCC | 12.9.86 (`/usr/local/cuda-12.9/bin/nvcc`) |
| GCC/G++ | 11.4.0 |
| CMake | 3.22.1 |
| Nsight Systems | 2025.1.3.140-251335620677v0 |
| Nsight Compute | 2025.2.1.0 installed at `/opt/nvidia/nsight-compute/2025.2.1/ncu`, not on `PATH`; performance-counter access denied |

The CMake/libcurl warning about `/usr/local/lib/libcurl.so.4` appeared during
configuration and testing; it is a pre-existing environment warning and did
not prevent builds or tests.

## Measurement protocol

All sweeps use Release builds, normalized deterministic synthetic FP32 data,
`Q=4`, `K=10`, seed `42`, two warm-up searches, and five measured searches.
Dataset generation is excluded. CPU E2E is the complete CPU backend call;
CUDA E2E is the complete uninstrumented `cuda-naive` backend call; CUDA kernel
latency is the existing CUDA Event measurement around only the similarity
kernel.

The reproducible runner is [`scripts/run_m2_benchmarks.py`](../scripts/run_m2_benchmarks.py):

```bash
python3 scripts/run_m2_benchmarks.py --binary build-cuda/vector_search
```

## Stage timing: where CUDA E2E time goes

`--timing-breakdown` is opt-in. CUDA Events measure the existing synchronous
H2D database copy, H2D query copy, kernel, and D2H score copy. Host
`steady_clock` measures the three allocation calls, CPU Top-K/score slicing,
cleanup, and the fully synchronized backend call. Cleanup includes event
destruction and `cudaFree`.

The total includes device selection/properties, event creation, host vector
allocation, and API overhead that are not individual rows. Therefore the
component sum is diagnostic and is not artificially adjusted to equal E2E.

| Workload | Allocation | H2D DB | H2D Query | Kernel | D2H Scores | CPU Top-K | Cleanup | Component sum | Total E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `N=10,000,D=128` | 0.481 | 0.613 | 0.058 | 0.195 | 0.074 | 0.275 | 0.361 | 2.057 | 2.246 |
| `N=100,000,D=128` | 0.837 | 4.534 | 0.129 | 3.523 | 0.287 | 2.481 | 0.656 | 12.447 | 13.885 |
| `N=100,000,D=256` | 1.320 | 10.860 | 0.130 | 18.909 | 0.486 | 2.202 | 1.057 | 34.964 | 37.760 |
| `N=100,000,D=768` | 1.694 | 31.888 | 0.116 | 64.144 | 0.612 | 2.398 | 2.149 | 103.001 | 110.006 |

The uninstrumented E2E for the 10K case was 2.030 ms in the crossover run;
the instrumented backend reported 2.247 ms. This overhead is expected for a
small workload and is why the crossover/dimension tables below use normal
benchmark mode.

## Crossover analysis

| N | CPU E2E | CUDA kernel | CUDA E2E | Speedup (CPU/CUDA) |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1.735 | 0.198 | 2.030 | 0.855x |
| 15,000 | 2.539 | 0.413 | 2.462 | 1.031x |
| 20,000 | 3.377 | 0.607 | 3.317 | 1.018x |
| 40,000 | 7.384 | 1.288 | 6.278 | 1.176x |
| 60,000 | 12.067 | 1.998 | 8.753 | 1.379x |
| 80,000 | 15.394 | 2.731 | 11.016 | 1.397x |
| 100,000 | 20.687 | 3.526 | 13.944 | 1.484x |

Observation: CPU is faster at 10K, while CUDA is faster at 15K and above in
this run. The sign change brackets the crossover between 10K and 15K vectors;
linear interpolation gives roughly 14K, but the small 15K margin means this is
an approximate, machine-specific boundary.

Inference supported by stage timing: the 10K kernel is only 0.195 ms, while
allocation, transfers, CPU Top-K, cleanup, and unlisted E2E overhead are much
larger. At 100K×128, the kernel is 3.523 ms and database H2D plus CPU Top-K
are 7.015 ms, so useful work amortizes per-search overhead. The measurements
do not isolate kernel-launch overhead as its own stage.

## Dimension analysis

Normal benchmark mode at `N=100,000`, `Q=4`, `K=10`:

| D | CPU E2E | CUDA kernel | CUDA E2E | Speedup (CPU/CUDA) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 21.952 | 3.189 | 13.784 | 1.593x |
| 256 | 41.090 | 18.842 | 36.366 | 1.130x |
| 768 | 141.561 | 64.300 | 107.664 | 1.315x |

Instrumented stage timing at the same N:

| D | H2D DB | Kernel | D2H Scores | CPU Top-K | Total E2E |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 4.534 | 3.523 | 0.287 | 2.481 | 13.885 |
| 256 | 10.860 | 18.909 | 0.486 | 2.202 | 37.760 |
| 768 | 31.888 | 64.144 | 0.612 | 2.398 | 110.006 |

Observation: increasing D materially raises both transfer and kernel stages.
The D2H score array has the same byte size for all three rows because N and Q
are fixed. Kernel latency rises about 5.9x from D=128 to D=256 and about 3.4x
from D=256 to D=768; this is not proportional to the dimension change.

Inference: the data supports both larger transfer volume and more dot-product
work as D grows. It does not establish whether the non-linear kernel behavior
comes from global-load efficiency, cache behavior, instruction throughput,
occupancy, or another GPU effect because Nsight Compute was unavailable.

## Nsight Systems

The reproducible wrapper is [`scripts/profile_nsys.sh`](../scripts/profile_nsys.sh):

```bash
scripts/profile_nsys.sh build-cuda/vector_search /tmp/vector-search-m2-nsys
```

It captured both representative workloads with `--trace=cuda`. The API trace
showed three backend searches per report (two warm-ups plus one measured run),
with nine `cudaMalloc`, nine `cudaMemcpy`, three kernel-launch, and nine
`cudaFree` calls. This is consistent with three device buffers per search, two
H2D copies plus one D2H copy, one kernel launch, and cleanup per search. The
operations are serialized; M1 has no streams or overlapping transfers.

The generated summaries did not contain CUDA GPU kernel or memory activity:

```text
SKIPPED: ... does not contain CUDA kernel data.
SKIPPED: ... does not contain GPU memory data.
```

An additional `--trace=cuda,cuda-hw` attempt returned:

```text
Illegal --trace argument 'cuda-hw'
One or more GPUs cannot be configured to collect a CUDA hardware trace.
```

Consequently, Systems provides API sequencing/call-count evidence here, but
not device-side busy/idle, GPU memory-transfer, or kernel-duration evidence.
The stage table uses CUDA Events and host timing for those quantities.

## Nsight Compute

The focused wrapper is [`scripts/profile_ncu.sh`](../scripts/profile_ncu.sh).
Although `which ncu` is empty, the existing Nsight Compute 2025.2.1.0 binary
is installed at `/opt/nvidia/nsight-compute/2025.2.1/ncu`; the wrapper uses it
without changing `PATH`. The profile attempt returned:

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

NVIDIA GPU Performance Counter access may need to be enabled on the Windows
host. No system package was installed, no privileged workaround was used, and
Windows NVIDIA Control Panel settings were not changed. The permission error
blocks profiler evidence for achieved occupancy, SM utilization, global-load
efficiency, device memory throughput, compute throughput, instruction/warp
behavior, and profiler-reported register usage.

For context only, a non-profiler `nvcc -Xptxas=-v` compile for `sm_86` reported
22 registers/thread, no barriers, and no stack frame or spills. The frozen
launch is 256 threads/block, with `ceil(Q*N/256)` equal to 157 blocks at 10K
and 1,563 blocks at 100K for Q=4. These facts do not classify the kernel as
memory-bound or compute-bound.

## M2.5 — Nsight Compute Analysis

M2.5 completed the profiler work that M2 could not collect because of
`ERR_NVGPUCTRPERM`. GPU performance counters are now available. The M1
`cuda-naive` implementation remains unchanged; this milestone adds focused
profiler evidence and analysis only.

### Environment and validation

The repository was verified at
`/home/sungj/projects/CUDA-Accelerated-Vector-Search-Engine` inside WSL2, not
under `/mnt/d`. The primary profile used Release
`N=100,000,D=768,Q=4,K=10`, seed 42, two warm-ups, and one profiled
launch. The optional comparison used `D=128`. GPU driver 576.88,
CUDA/NVCC 12.9/12.9.86, and Nsight Compute 2025.2.1.0 were verified.

The permission probe and both profiles completed without
`ERR_NVGPUCTRPERM`. Reports and 20–24 KB CSV summaries are under
`profiling/ncu/`. The reproducible wrapper is
[scripts/profile_ncu.sh](../scripts/profile_ncu.sh).

Release CPU and CUDA builds were configured for `sm_86` and built successfully.
CPU CTest passed 1/1; CUDA CTest passed 2/2, including CUDA correctness. A
normal `cuda-naive` smoke search completed; the 10K×128 CPU/CUDA
checksums were `117359.130` and `117359.128`.

### Launch configuration

For D=768, `Q*N=400,000`, so the unchanged launch was
`grid=(1,563,1)`, `block=(256,1,1)`, and 400,128 threads. Nsight
Compute reported 28 registers/thread, 0 static/dynamic shared memory, 1.02
KiB driver-reserved shared memory, 100% theoretical occupancy, and 95.43%
achieved occupancy (45.81 active warps/SM out of 48 theoretical). Block
limits were SM=16, registers=8, shared memory=8, and warps=6. The direct
`nvcc -Xptxas=-v` diagnostic reported 22 registers, zero barriers, and
zero spills; the runtime profiler launch value is 28.
The same launch at D=128 retained 28 registers/thread and 100% theoretical
occupancy, with 85.67% achieved occupancy and 41.12 active warps/SM.

### Speed of light and memory

| Metric | D=768 | D=128 |
| --- | ---: | ---: |
| Compute (SM) throughput | 0.26% peak | 6.39% |
| Memory/DRAM throughput | 27.19% peak | 58.14% |
| DRAM rate | 47.81 GB/s | 102.24 GB/s |
| Memory busy | 13.38% | 52.35% |
| L1/TEX hit rate | 73.91% | 76.24% |
| L2 hit rate | 11.33% | 13.15% |
| Nsight Compute device duration | 55.42 ms | 3.46 ms |

Replay-normalized utilization varied slightly between focused passes, but
D=768 consistently had low compute and sub-peak memory utilization. The
Speed-of-Light guidance therefore points to latency exposure, not saturated
compute or DRAM bandwidth. Normal unprofiled validation was 60.130 ms
kernel/106.778 ms E2E at D=768 and 3.493 ms kernel/14.080 ms E2E at D=128.

### Global-memory evidence

Global loads averaged 4.0 useful bytes per 32-byte sector. Nsight Compute
explicitly flagged that only 4/32 transmitted bytes were used per thread and
warned of a stride between threads. The same 4/32 result appeared at D=128;
the D=768 report noted 87.3% of relevant sectors missed in L2.

At a fixed dimension, adjacent threads access database rows separated by
`768*4 = 3,072` bytes. The measured sector utilization therefore matches
the row-major one-thread-per-vector mapping. A high L1/TEX hit rate does not
disprove this because query values are reused; it can coexist with poor
database-sector use.

### Warp evidence

At D=768, schedulers had 11.11 active warps but only 0.05 eligible warps per
scheduler. Only 2.01% of cycles had an eligible warp; 97.99% had none, and
the issued-warp rate was 0.02. Warp cycles per issued instruction were
551.48. Dominant stalls were LG-memory instruction-queue throttle (333.4
cycles, 60.46%) and L1TEX scoreboard dependency (219.0 cycles, 39.71%).
D=128 showed the same pattern: 96.58% no-eligible cycles, 0.07 eligible
warps, and about 301 warp cycles per issued instruction. Source-level warp
sampling was unavailable, so no source-line attribution is claimed.

### D comparison and hypotheses

D=768 executed 77,612,532 instructions versus 13,612,532 at D=128, and warp
latency rose from about 301 to 551 cycles. Thus:

* **Strided global-memory access — confirmed.** The 4/32 sector metric,
  explicit stride warning, low L2 hit rate, and memory queue/L1TEX stalls
  support inefficient global-load transactions. This is not a claim that
  DRAM bandwidth is saturated.
* **Sequential dimension processing — partially supported.** Instruction
  count and warp latency grow sharply with D, making dimension-parallel work
  justified. The profile does not isolate scalar-accumulator dependency as
  the main cause; memory stalls dominate.

### M3 recommendation

Use Candidate A as the single primary experiment: a separate block-per-
query/vector backend with threads cooperating across dimensions and a
block-level reduction. It directly tests adjacent-dimension coalescing and
removes the long serial D loop. Expected costs are synchronization,
reduction instructions, many blocks, and changed resource usage. Validate
with the same correctness checks, kernel/E2E timings, sector utilization,
occupancy, and warp metrics.

Candidate B (dimension-major/transposed database) is a secondary coalescing
experiment with preprocessing/layout and H2D costs. Candidate C
(warp/subgroup-oriented multiple-vectors-per-block) is a secondary mapping
experiment with subgroup-tail and reduction choices. Persistent database
residency remains separate: M2 measured about 29% of E2E in database H2D at
100K×768 and it must not be combined with the first kernel experiment.

No M3 kernel or system optimization was implemented in M2.5.

## Confirmed bottlenecks (M2 stage timing)

1. At 100K×768, the similarity kernel is the largest measured stage: 64.144
   ms, 58.3% of instrumented E2E.
2. Database H2D is the largest non-kernel stage at 100K×768: 31.888 ms,
   29.0%; it is also 32.7% at 100K×128.
3. At 10K×128, allocation plus cleanup is 0.842 ms and CPU Top-K is 0.275
   ms while the kernel is 0.195 ms, confirming that small workloads are
   dominated by non-kernel work and measurement overhead.

The full score D2H transfer is not dominant in the large case: it is 0.612 ms
(about 0.6%) at 100K×768. At the M2-only stage, the row-major/adjacent-thread
strided-access hypothesis was still unconfirmed; M2.5 later confirmed it with
Nsight Compute and M3 tests the resulting redesign below.

## M3 recommendation (historical M2 record)

M2 provisionally recommended a separate experimental backend with a
global-memory-friendly thread/data mapping and deferred the exact mapping
until profiler evidence was available. The current evidence-backed decision
is the M2.5 Candidate A recommendation above.

Secondary candidates are database residency/transfer reduction, GPU-side Top-K
or score-transfer reduction, and reuse of device allocations. None of these
was implemented in M2.

# M3 — Block-per-Vector Parallel Reduction

## Experimental setup

M3 adds cuda-block as a separate backend and leaves cuda-naive unchanged. Both
backends use the same SearchRequest, device allocation lifecycle, synchronous
H2D database/query copies, score-array D2H copy, CPU Top-K, cleanup, and CUDA
Event timing boundaries. The only intended kernel change is the mapping:

~~~text
cuda-naive: one thread -> one complete query/database-vector dot product
cuda-block: one 256-thread block -> one pair; threads -> dimensions
~~~

The block kernel uses one explicit shared-memory tree reduction. It does not
use warp shuffles, vectorized loads, query caching, a transposed layout,
streams, pinned memory, persistent device data, or GPU Top-K.

The repository was validated inside WSL2 at
/home/sungj/projects/CUDA-Accelerated-Vector-Search-Engine. The Release
builds used CUDA 12.9, NVCC 12.9.86, GCC 11.4.0, and
CMAKE_CUDA_ARCHITECTURES=86. CPU CTest passed 1/1 and CUDA CTest passed 2/2.
The CUDA tests cover D=3, 127, 128, 255, 256, 257, and 768, multiple
queries, K=1, K>1, and tolerance-based CPU/naive/block comparison.

## Normal benchmark comparison

The primary configuration is N=100,000, D=768, Q=4, K=10, seed 42, two
warm-ups, and five measured iterations in a Release build. The same protocol
was used for the D sweep and small-workload check. Speedups are naive divided
by block.

| Workload | CPU E2E (ms) | Naive kernel (ms) | Block kernel (ms) | Kernel speedup | Naive E2E (ms) | Block E2E (ms) | E2E speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary N=100,000, D=768 | 162.050 | 63.835 | 7.993 | 7.986x | 111.474 | 56.139 | 1.986x |
| N=100,000, D=128 | 24.470 | 3.544 | 5.476 | 0.647x | 13.969 | 16.459 | 0.849x |
| N=100,000, D=256 | 48.888 | 18.923 | 5.637 | 3.357x | 37.082 | 23.640 | 1.569x |
| N=100,000, D=768 | 151.464 | 63.524 | 7.994 | 7.946x | 111.358 | 52.090 | 2.138x |
| Small N=10,000, D=128 | 1.866 | 0.184 | 0.539 | 0.341x | 1.864 | 2.549 | 0.731x |

The machine-readable records are
[m3_primary.csv](../benchmarks/results/m3_primary.csv),
[m3_dimension.csv](../benchmarks/results/m3_dimension.csv),
[m3_small_workload.csv](../benchmarks/results/m3_small_workload.csv), and
[m3_comparison.csv](../benchmarks/results/m3_comparison.csv).

## Nsight Compute comparison

The focused M2.5 sections were reused for both kernels on the primary
workload. The direct metric CSVs add the global-memory bytes-per-sector and
LG/L1TEX stall counters. The section report values are:

| Metric | cuda-naive | cuda-block | Change |
| --- | ---: | ---: | --- |
| Global-load useful bytes/32-byte sector | 4 (12.5%) | 32 (100%) | 8x better |
| Global-store useful bytes/32-byte sector | 32 (100%) | 4 (12.5%) | New store inefficiency |
| DRAM throughput | 51.41 GB/s (28.23%) | 125.76 GB/s (70.21%) | Higher |
| L1/TEX hit rate | 71.75% | 49.91% | Lower |
| L2 hit rate | 14.17% | 2.27% | Lower |
| SM throughput | 2.55% | 74.74% | Higher |
| Memory throughput | 28.23% | 74.74% | Higher |
| Theoretical / achieved occupancy | 100% / 95.46% | 100% / 93.93% | Achieved -1.53 pp |
| Active warps/SM | 45.82 | 45.09 | Essentially unchanged |
| Registers/thread | 28 | 24 | Lower |
| Static shared memory/block | 0 B | 1.02 KiB | Reduction state |
| Executed instructions | 77,618,984 | 632,800,000 | Higher from reduction/barriers |
| Eligible warps/scheduler | 0.03 | 1.31 | Higher |
| No-eligible-warp cycles | 98.67% | 41.01% | Lower |
| LG-throttle direct stall share | 63.47% | 0.00% | Lower |
| L1TEX long-scoreboard direct stall share | 35.71% | 38.94% | Similar share |
| L1TEX scoreboard rule stall | 310.7 cycles | 7.4 cycles | Lower absolute wait |
| Warp cycles/issued instruction | 860.95 | 19.02 | Lower |
| Nsight Compute kernel duration | 54.01 ms | 10.12 ms | Lower |

The block report's remaining MemoryWorkloadAnalysis rule is for global stores:
one thread per block writes one score, so that output is inherently sparse
within a 256-thread block. This does not negate the direct global-load result.
L2 hit rate also falls, so M3 should be described as a coalescing and
dimension-parallelism experiment, not an L2-locality optimization.

Reports and direct metric summaries are under
[profiling/ncu/m3](../profiling/ncu/m3/). The wrapper is
[scripts/profile_ncu.sh](../scripts/profile_ncu.sh); it accepts either
cuda-naive or cuda-block while keeping the same focused sections.

## Hypothesis Evaluation

### Global-Memory Coalescing

Confirmed improvement for the targeted loads. Direct Nsight Compute data
increased useful global-load bytes per sector from 4 to 32, matching the
expected adjacent-dimension access pattern. The tradeoff is a new 4-byte
per-sector score-store pattern because only thread 0 writes each block's
score, and L2 hit rate fell from 14.17% to 2.27%.

### Memory-Latency Hiding

Improved. Eligible warps increased from 0.03 to 1.31 per scheduler and
no-eligible cycles fell from 98.67% to 41.01%. Direct LG-throttle share fell
from 63.47% to 0%; the L1TEX scoreboard share remained similar, but its
absolute rule stall estimate fell from 310.7 to 7.4 cycles and warp
cycles/issued instruction fell from 860.95 to 19.02.

### Parallel Dimension Processing

Beneficial for D=256 and D=768, where block kernel latency improved 3.36x and
7.95x respectively, and primary E2E latency improved 1.99x. Harmful for
D=128 and the 10K check, where fixed-block synchronization and reduction
overhead outweighed the smaller amount of dimension work.

## M3 Result

**successful**

M3 is successful as a profiler-driven experiment: it directly corrected the
confirmed global-load inefficiency and materially reduced high-D kernel and
E2E latency, while exposing measurable synchronization, store-sector, and
low-D overhead tradeoffs. cuda-naive remains the frozen M1 baseline and
cuda-block remains available for direct comparison.

## Next Recommendation

Recommend exactly one M4 experiment: warp-per-vector reduction. It should
retain the now-proven adjacent-dimension load mapping while reducing the
full-block shared-memory/barrier overhead that caused the D=128 and small
workload regressions. Do not implement it as part of M3.

# M4 — Warp-per-Vector Reduction

## Progression

The measured implementation path is now:

~~~text
M0 CPU
  ->
M1 thread-per-vector cuda-naive
  ->
M2/M2.5 profiling
  ->
M3 block-per-vector cuda-block
  ->
M4 warp-per-vector cuda-warp
~~~

M4 is a separate experimental backend. cuda-naive and cuda-block remain
available and unchanged as algorithmic baselines.

## Design and Frozen Pipeline

The first M4 experiment uses 256 threads per block and eight warps per block.
Each warp handles one query/database-vector pair, each lane processes
dimensions lane, lane+32, lane+64, and so on, and five synchronized
shuffle-down steps produce the dot product in lane 0. A ballot-derived mask
handles a partially used final block. The kernel has zero explicit
shared-memory reduction state and no block-wide barrier.

The host data flow is unchanged:

~~~text
allocation -> synchronous H2D database -> synchronous H2D queries
-> similarity kernel -> synchronous D2H all scores -> CPU Top-K -> cleanup
~~~

No persistent device data, allocation reuse, pinned memory, streams,
asynchronous copies, vectorized loads, layout transformation, or GPU Top-K
was included.

## Backend Comparison

The primary protocol is N=100,000, Q=4, K=10, seed 42, Release, two
warm-ups, and five measured iterations. Kernel values are CUDA Event
measurements; E2E includes the unchanged allocation, copies, kernel, D2H,
CPU Top-K, and cleanup pipeline.

| Workload | Naive kernel | Block kernel | Warp kernel | Warp vs naive | Warp vs block | Naive E2E | Block E2E | Warp E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N=100,000, D=128 | 3.388 ms | 5.712 ms | 1.123 ms | 3.017x | 5.086x | 14.291 ms | 16.460 ms | 11.562 ms |
| N=100,000, D=256 | 19.149 ms | 5.559 ms | 2.351 ms | 8.145x | 2.365x | 37.141 ms | 23.180 ms | 19.898 ms |
| N=100,000, D=768 | 64.447 ms | 7.879 ms | 6.918 ms | 9.316x | 1.139x | 108.782 ms | 51.515 ms | 50.381 ms |
| N=10,000, D=128 | 0.202 ms | 0.530 ms | 0.123 ms | 1.642x | 4.309x | 2.345 ms | 2.547 ms | 2.050 ms |

At the primary D=768 workload, warp improved kernel latency 1.139x over
block and E2E latency 1.023x over block. At D=128 it improved kernel and E2E
by 5.086x and 1.424x over block. The small-workload E2E regression from M3
was reduced: warp was 2.050 ms versus block at 2.547 ms, although it remained
slower than the 1.926 ms CPU E2E result.

Machine-readable records are
[m4_primary.csv](../benchmarks/results/m4_primary.csv),
[m4_dimension.csv](../benchmarks/results/m4_dimension.csv),
[m4_small_workload.csv](../benchmarks/results/m4_small_workload.csv), and
[m4_comparison.csv](../benchmarks/results/m4_comparison.csv).

## Dimension Sweep

For N=100,000, Q=4, K=10, warp was the fastest kernel at D=64, 128, 256,
512, 768, and 1024. Naive was fastest at D=32. Warp was the fastest E2E
backend at D=64, 128, 256, 512, and 1024; block was marginally faster than
warp at D=768 E2E.

| D | Naive kernel | Block kernel | Warp kernel | Best kernel | Naive E2E | Block E2E | Warp E2E | Best E2E |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 32 | 0.521 ms | 5.347 ms | 0.888 ms | naive | 6.511 ms | 11.129 ms | 6.902 ms | naive |
| 64 | 1.378 ms | 6.132 ms | 0.726 ms | warp | 8.677 ms | 13.864 ms | 7.938 ms | warp |
| 128 | 3.388 ms | 5.712 ms | 1.123 ms | warp | 14.291 ms | 16.460 ms | 11.562 ms | warp |
| 256 | 19.149 ms | 5.559 ms | 2.351 ms | warp | 37.141 ms | 23.180 ms | 19.898 ms | warp |
| 512 | 43.011 ms | 6.815 ms | 4.550 ms | warp | 73.915 ms | 37.670 ms | 35.529 ms | warp |
| 768 | 63.601 ms | 7.975 ms | 6.985 ms | warp | 109.778 ms | 51.193 ms | 51.987 ms | block |
| 1024 | 88.739 ms | 10.016 ms | 9.268 ms | warp | 146.875 ms | 68.729 ms | 66.828 ms | warp |

## Nsight Compute Comparison

The M4 profiles reused the focused M2.5/M3 sections at D=768 and D=128.
They are stored under [profiling/ncu/m4](../profiling/ncu/m4/).

| Metric | D=768 block | D=768 warp | D=128 block | D=128 warp |
| --- | ---: | ---: | ---: | ---: |
| Useful global-load bytes/32-byte sector | 32 (100%) | 32 (100%) | 32 (100%) | 32 (100%) |
| DRAM memory throughput | 125.00 Gbyte/s (70.22%) | 177.48 Gbyte/s (97.09%) | 31.16 Gbyte/s (17.06%) | 185.12 Gbyte/s (96.60%) |
| L1/TEX hit rate | 49.91% | 50.18% | 49.61% | 50.86% |
| L2 hit rate | 2.24% | 2.23% | 12.00% | 5.24% |
| Achieved occupancy | 93.95% | 89.31% | 91.32% | 86.24% |
| Registers/thread | 24 | 38 | 24 | 38 |
| Static shared memory/block | 1.02 KiB | 0 B | 1.02 KiB | 0 B |
| Eligible warps/scheduler | 1.31 | 0.39 | 1.18 | 1.35 |
| No-eligible cycles | 41.00% | 73.72% | 44.07% | 38.85% |
| Warp cycles/issued instruction | 19.01 | 40.98 | 19.45 | 15.91 |
| Dominant stall | L1TEX scoreboard, 7.4 cycles | L1TEX scoreboard, 36.7 cycles | CTA barrier, 6.6 cycles | L1TEX scoreboard, 8.3 cycles |

Warp preserved M3's coalesced loads at both dimensions and removed the static
shared-memory reduction array. At D=128, executed instructions dropped from
444,000,000 to 80,400,000 and no-eligible cycles fell from 44.07% to 38.85%,
supporting the reduction-overhead hypothesis. At D=768, executed
instructions dropped from 632,800,000 to 202,403,225, but higher register
use and fewer eligible warps show a latency-hiding tradeoff. The profiler
evidence therefore supports a smaller reduction scope without implying that
all scheduler metrics improve.

## Hypothesis Evaluation

### Coalescing

Preserved. M4 measured 32 useful bytes per 32-byte global-load sector, the
same result as M3, at both D=128 and D=768.

### Small-D Reduction Overhead

Improved. At D=128, warp was 5.086x faster than block in kernel time and
1.424x faster E2E. The 10K D=128 E2E regression fell from 2.547 ms for block
to 2.050 ms for warp. The D=128 profile also removed the CTA-barrier stall
identified for block and reduced executed instructions.

### High-D Performance

Competitive, not uniformly dominant. Warp was 1.139x faster than block in
the primary D=768 kernel measurement and 1.023x faster E2E, but block was
slightly faster in the independent D=768 sweep E2E row. Warp remained the
fastest kernel through D=1024 in that sweep.

### Reduction Cost

Improved in reduction state and instruction count: static shared memory fell
from 1.02 KiB to zero, and the warp reduction uses five synchronized shuffle
steps without block-wide barriers. The D=768 scheduler profile is a
tradeoff: warp has higher no-eligible cycles and L1TEX scoreboard stalls,
likely reflecting its higher register use and different number of active
pair blocks. This is why reduction improvement is supported by the
implementation and D=128 profiler evidence, rather than inferred from
instruction count alone.

## M4 Result

**successful**

M4 successfully preserved M3 coalescing, improved the primary D=128 and
small-workload regressions, and remained competitive at D=768 while keeping
the frozen host pipeline. It is an experimental backend, not an automatic
selection policy.

## M5 Recommendation

Run one controlled experiment: sweep the number of warps per block/block
size for the warp-per-vector mapping while holding transfers, allocation,
CPU Top-K, and benchmark methodology fixed. The purpose is to test whether
the D=768 scheduler tradeoff is caused by the mandated eight-warps/block
configuration. Do not implement this recommendation as part of M4.


# M5 — GPU-Resident Database

## Progression

The project narrative is now:

~~~text
M0 CPU baseline
  ->
M1 naive CUDA
  ->
M2 latency decomposition
  ->
M2.5 memory profiling
  ->
M3 block-per-vector
  ->
M4 warp-per-vector
  ->
M5 GPU-resident database
~~~

M0–M4 are primarily correctness, measurement, and kernel-optimization
milestones. M5 is the first system-level optimization: it preserves the M4
similarity kernel and changes only the database allocation/upload lifetime.

## M4 Reconciliation

The M4 primary D=768 row and the dimension-sweep D=768 row were not the same
measurement. The primary run recorded cuda-block at 7.879 ms kernel and
51.515 ms E2E, and cuda-warp at 6.918 ms kernel and 50.381 ms E2E. The
independent dimension-sweep row recorded cuda-block at 7.975 ms kernel and
51.193 ms E2E, and cuda-warp at 6.985 ms kernel and 51.987 ms E2E.

The M4 dimension-sweep runner starts separate backend subprocesses for each
row. The 0.794 ms E2E difference is about 1.5% and is consistent with
ordinary run-to-run variance. The summary correctly selected cuda-block for
the independent sweep row. No M4 CSV data was changed and no summary-script
bug was found.

## Implementation

cuda-warp remains the stateless M4 baseline. Every search still allocates
and uploads the database, uploads queries, runs the unchanged M4
warp-per-vector kernel, copies scores back, runs CPU Top-K, and cleans up.

cuda-warp-resident adds ResidentSearchBackend with this lifecycle:

~~~text
create backend
  -> prepare_database(database)
  -> search(query batch 1)
  -> search(query batch 2)
  -> search(query batch 3)
  -> ...
  -> reload_database(database) or clear_database()
  -> destroy backend
~~~

prepare_database allocates and uploads the FP32 database once and keeps the
device allocation alive. Repeating preparation for the same host pointer, N,
and D is idempotent. reload_database explicitly reuploads, including for the
same host pointer. Changed identity or dimensions require a new preparation.
search validates the prepared dimensions and can receive a null database
pointer after preparation; query and score buffers remain per-search
allocations for a fair M4 comparison. clear_database and RAII destructors
release the device database.

Preparation timing is separate from query timing:

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

The CLI's repeated-batch mode generates one deterministic database and
different query contents for each sequential batch. Dataset generation is
excluded from all search latency measurements.

## Database Preparation

The formal configuration was N=100,000, Q=4, K=10, seed=42, Release, two
warmups, and 100 measured batches. The optional workload used N=250,000.
Database memory is N * D * sizeof(float).

| Workload | Database bytes | Database MiB | DB allocation (ms) | DB H2D (ms) | Prepare total (ms) | Free before (bytes) | Free after (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary N=100,000, D=768 | 307,200,000 | 292.969 | 1.070 | 367.040 | 660.766 | 3,449,500,468 | 3,141,219,124 |
| N=100K, D=128 | 51,200,000 | 48.828 | 0.277 | 58.707 | 362.981 | 3,449,500,468 | 3,397,071,668 |
| N=100K, D=256 | 102,400,000 | 97.656 | 0.429 | 121.751 | 432.986 | 3,449,500,468 | 3,346,740,020 |
| N=100K, D=768 | 307,200,000 | 292.969 | 1.794 | 368.860 | 683.011 | 3,449,500,468 | 3,141,219,124 |
| N=250K, D=768 | 768,000,000 | 732.422 | 3.075 | 921.983 | 1,242.559 | 3,449,500,468 | 2,679,845,684 |

The 250K database fit on the 4 GiB RTX 3050 Ti Laptop GPU. The backend
failed explicitly on allocation/copy errors; it did not fall back to CPU.

## Cold-Start Performance

Cold resident latency is database preparation plus the first query. These
one-shot runs used warmup=0 and one measured batch.

| Workload | cuda-warp E2E | Resident prepare | Resident first query | Resident cold total | Cold ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary N=100K, D=768 | 526.642 ms | 660.766 ms | 15.873 ms | 676.639 ms | 0.778x |
| N=100K, D=128 | 168.031 ms | 362.981 ms | 9.331 ms | 372.313 ms | 0.451x |
| N=100K, D=256 | 233.638 ms | 432.986 ms | 9.934 ms | 442.921 ms | 0.527x |
| N=100K, D=768 | 496.184 ms | 683.011 ms | 15.672 ms | 698.683 ms | 0.710x |
| N=250K, D=768 | 1,085.765 ms | 1,242.559 ms | 37.961 ms | 1,280.520 ms | 0.848x |

Cold ratio is cuda-warp E2E divided by resident cold total. It is below
1.0x for every workload, so database residency is a regression for one-shot
requests. The preparation total includes process-level CUDA initialization
observed by the host timer; it is intentionally not hidden inside warm query
latency.
The final formal run retained its raw process-level CUDA initialization and
transfer timing; the paired within-run comparison is the meaningful result
when host-to-device timing varies across process runs.

## Warm Performance

Warm means are over 100 measured sequential query batches after two warmups.
Database preparation is excluded.

| Workload | cuda-warp mean | Resident mean | Speedup | cuda-warp median | Resident median |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary N=100K, D=768 | 395.504 ms | 13.289 ms | 29.761x | 395.605 ms | 13.391 ms |
| N=100K, D=128 | 72.177 ms | 6.986 ms | 10.331x | 72.042 ms | 6.966 ms |
| N=100K, D=256 | 136.776 ms | 8.278 ms | 16.523x | 136.710 ms | 8.384 ms |
| N=100K, D=768 | 394.634 ms | 13.437 ms | 29.369x | 394.637 ms | 13.341 ms |
| N=250K, D=768 | 986.258 ms | 33.958 ms | 29.043x | 985.571 ms | 33.804 ms |

For the primary M5 repeated run, the same-run values were 395.504 ms stateless
and 13.289 ms resident, a 29.761x speedup. Relative to the recorded M4
primary cuda-warp E2E of 50.381 ms, resident warm latency is 73.6% lower.
The controlled M5 stateless-to-resident reduction is 96.6%. The stateless
database H2D stage averaged 369.096 ms, which was 93.3% of its 395.502 ms
stage-timed E2E. The earlier M2 sample was about 31.9 ms; this final run
retained the raw environment-sensitive transfer timing. Removing database
allocation and cleanup adds savings beyond the H2D stage.

## Amortization

The resident total includes the one-time preparation from the repeated run.
The stateless total is the sum of the same number of measured cuda-warp
batches.

| Query batches | Stateless total | Stateless avg | Resident total incl. prepare | Resident avg incl. prepare | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 396.543 ms | 396.543 ms | 687.126 ms | 687.126 ms | 0.577x |
| 2 | 792.198 ms | 396.099 ms | 700.819 ms | 350.410 ms | 1.130x |
| 5 | 1,978.387 ms | 395.677 ms | 742.382 ms | 148.476 ms | 2.665x |
| 10 | 3,953.803 ms | 395.380 ms | 807.001 ms | 80.700 ms | 4.899x |
| 20 | 7,911.911 ms | 395.596 ms | 938.998 ms | 46.950 ms | 8.426x |
| 100 | 39,550.352 ms | 395.504 ms | 2,001.565 ms | 20.016 ms | 19.760x |

The algebraic crossover estimated from the primary warm means is about 1.8
batches, with the first requested table point showing a benefit at 2.
Crossover estimates were about 5.3 batches at D=128, 3.1 at D=256, 1.7 at
D=768, and 1.4 at N=250K, D=768. The first requested benefit points were
10, 5, 2, and 2 batches respectively. These are estimates from the
repeated-run preparation sample, not the separate one-shot cold sample.

## Stage Timing

Primary N=100K, D=768 per-batch means:

| Stage | Stateless cuda-warp | Resident warm |
| --- | ---: | ---: |
| Per-search device allocation | 1.570 ms (DB/query/score) | 0 (DB; query below) |
| Database H2D | 369.096 ms | 0 (preparation only) |
| Query allocation | included in allocation | 0.401 ms |
| Query H2D | 0.265 ms | 0.185 ms |
| Kernel | 7.459 ms | 7.038 ms |
| Score D2H | 2.321 ms | 2.136 ms |
| CPU Top-K | 2.415 ms | 2.452 ms |
| Cleanup | 3.069 ms | 0.433 ms |
| Stage E2E | 395.502 ms | 13.287 ms |

Resident preparation in the repeated run was 672.651 ms, with DB allocation
1.010 ms and DB H2D 369.732 ms. Kernel latency remained effectively
unchanged, confirming that the measured change is system-level database
lifetime rather than a kernel optimization.

## Nsight Systems

A representative Nsight Systems trace used N=10,000, D=128, Q=2, K=3,
three measured batches, zero warmups, and both backends. The trace wrapper is
scripts/profile_m5_nsys.sh. The reports are:

* profiling/nsys/m5/m5_cuda-warp_n10000_d128_q2.nsys-rep
* profiling/nsys/m5/m5_cuda-warp-resident_n10000_d128_q2.nsys-rep

The CUDA API summaries recorded:

| API | cuda-warp | cuda-warp-resident |
| --- | ---: | ---: |
| cudaMalloc | 9 | 7 |
| cudaMemcpy | 9 | 7 |
| cudaFree | 9 | 7 |
| cudaLaunchKernel | 3 | 3 |
| cudaMemGetInfo | 0 | 2 |

For stateless, the nine copies are three database H2D, three query H2D,
and three score D2H operations. For resident, the seven copies are one
database H2D during preparation, three query H2D, and three score D2H
operations. The API counts and application stage output therefore confirm
that repeated database H2D was eliminated. Nsight's GPU memory summary
tables were not populated by this WSL capture, but the CUDA API trace
recorded the lifecycle difference directly.

## New Bottleneck

For the primary D=768 resident workload, the unchanged similarity kernel
remains the largest steady-state component at 7.038 ms, 53.0% of the
13.289 ms query E2E. CPU Top-K is next at 2.452 ms, 18.5%. At D=128, CPU
Top-K becomes dominant at 2.481 ms versus a 1.158 ms kernel. M5 therefore
moves the system bottleneck away from database movement; the remaining
dominant stage depends on workload, with the kernel dominant for the primary
and CPU Top-K dominant for lower-dimensional queries.

## Result

**successful for the repeated-query objective; partially successful for
one-shot latency.**

The warm resident backend improved the primary workload by 29.761x and all
tested workloads by 10.331x–29.761x, including the optional 250K workload at
29.043x. Cold resident searches regressed to 0.451x–0.848x of stateless
one-shot latency because preparation is paid up front. The implementation
meets the M5 objective without changing cuda-naive, cuda-block, or
cuda-warp algorithmically and without implementing M6.

## M6 Recommendation

Run exactly one next experiment: a CPU Top-K scaling experiment after
database residency, sweeping Q and K at D=128, D=256, and D=768 while
holding the frozen M4 kernel and transfers unchanged. This quantifies
whether CPU Top-K is the next system bottleneck for lower-dimensional
resident workloads. Do not implement it as part of M5.
