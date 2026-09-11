# Performance Report — M0 through M5

This report records the measured optimization path of the exact FP32 vector-search engine. It separates kernel-only latency from end-to-end latency and treats the CPU implementation as the correctness reference. Raw measurements are retained under [`benchmarks/results/`](../benchmarks/results/README.md), while the experiment-by-experiment reasoning is preserved in [`optimization_log.md`](optimization_log.md).

## Executive Summary

The first CUDA kernel assigned one thread to each database-vector/query pair. That mapping exposed parallelism across pairs, but each warp loaded one scalar from widely separated database rows. At dimension 768, adjacent threads were approximately 3,072 bytes apart and each 32-byte global-load sector carried only 4 useful bytes.

Nsight Compute identified that access pattern as the principal kernel bottleneck. Reassigning a whole block, then a whole warp, to each vector pair made lanes cooperate across adjacent dimensions. The warp design preserved 32/32 useful bytes per global-load sector while avoiding the full-block synchronization and instruction overhead that penalized smaller dimensions.

For the primary `N=100,000`, `D=768`, `Q=4`, `K=10` workload:

| Backend | Kernel latency (ms) | Speedup vs `cuda-naive` |
|---|---:|---:|
| `cuda-naive` | 64.447 | 1.000x |
| `cuda-block` | 7.879 | 8.180x |
| `cuda-warp` | 6.918 | **9.316x** |

After kernel latency fell, repeated database host-to-device transfer became a system-level bottleneck. M5 therefore added a GPU-resident database lifecycle. It did not change the warp kernel and did not improve one-shot cold latency; its benefit is the removal of repeated database uploads across query batches.

## Tested Environment

| Component | Version |
|---|---|
| OS | WSL2, Ubuntu 22.04.4 LTS |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GiB VRAM |
| NVIDIA driver | 576.88 |
| CUDA toolkit | 12.9 |
| NVCC | 12.9.86 |
| GCC | 11.4.0 |
| CMake | 3.22.1 |
| Nsight Systems | 2025.1.3.140 |
| Nsight Compute | 2025.2.1.0 |

Measurements in different milestone CSVs were collected in separate runs. Small differences for nominally identical workloads are expected from normal run-to-run variance. Tables do not silently substitute values from one run into another.

## Measurement Definitions

- **Kernel-only latency** measures the timed CUDA similarity-kernel interval after launch and before completion. It excludes allocation, copies, CPU Top-K, and cleanup.
- **End-to-end latency** measures the complete backend search call.
- **Resident preparation** measures the one-time database allocation and upload.
- **Resident warm-query latency** measures a search after preparation, so it excludes the database upload by design.
- All benchmark runners use warm-up iterations followed by multiple measured iterations unless a cold-start experiment explicitly states otherwise.

## M0 — CPU Reference

### Observation

An exact, deterministic reference was required before GPU implementations could be evaluated.

### Hypothesis

A simple serial C++ implementation would provide the clearest correctness oracle even if it was not the performance target.

### Change

M0 implemented exact FP32 dot-product scoring followed by deterministic CPU Top-K ordering.

### Measurement

The CPU backend is included in the scaling and correctness datasets, but its primary role is reference behavior rather than a CUDA speedup headline.

### Conclusion

Every CUDA backend must match the CPU scores and indices within the configured floating-point tolerance. CPU Top-K remains intentionally shared so kernel work can be compared without changing ranking semantics.

## M1 — Naive CUDA Baseline

### Observation

The first GPU implementation mapped one CUDA thread to one database-vector/query pair and accumulated all dimensions serially.

### Hypothesis

Parallelizing the N × Q pair space would establish a functional CUDA baseline and expose which part of the implementation limited performance.

### Change

M1 added `cuda-naive`, including device allocation, database/query upload, kernel execution, score download, CPU Top-K, and cleanup.

### Measurement

For the later primary M4 comparison at `N=100,000`, `D=768`, `Q=4`, `K=10`, the naive kernel measured 64.447 ms.

### Conclusion

The implementation was correct, but the mapping left dimension-level parallelism unused and produced an unfavorable row-major memory-access pattern within each warp.

## M2 — Stage Timing and Bottleneck Analysis

### Observation

End-to-end latency alone could not distinguish kernel cost from allocation, data movement, Top-K, or cleanup.

### Hypothesis

Explicit stage timing would identify the dominant work and prevent optimization based on intuition alone.

### Change

M2 instrumented allocation, database H2D, query H2D, kernel, score D2H, CPU Top-K, cleanup, and total end-to-end latency.

### Measurement

The representative `N=100,000`, `D=768`, `Q=4`, `K=10` stage run reported:

| Stage | Latency (ms) | Share of E2E |
|---|---:|---:|
| Allocation | 1.694 | 1.5% |
| Database H2D | 31.888 | 29.0% |
| Query H2D | 0.116 | 0.1% |
| Kernel | 64.144 | 58.3% |
| Score D2H | 0.612 | 0.6% |
| CPU Top-K | 2.398 | 2.2% |
| Cleanup | 2.149 | 2.0% |
| End-to-end | 110.006 | 100% |

The individually timed stages do not sum exactly to end-to-end time because the latter also includes host-side orchestration and timing overhead.

### Conclusion

Kernel execution was the largest measured component, with database upload second. The evidence justified profiling and redesigning the kernel before pursuing Top-K or transfer optimizations.

## M2.5 — Nsight Compute Profiling

### Observation

Stage timing established that the kernel mattered but did not explain why it was slow.

### Hypothesis

The one-thread-per-pair mapping caused uncoalesced row-major loads and memory-dependency stalls.

### Change

M2.5 added reproducible Nsight Compute collection and retained concise metric summaries.

### Measurement

For the `D=768` naive kernel, Nsight Compute reported:

| Metric | `cuda-naive` |
|---|---:|
| Global-load useful bytes per 32-byte sector | **4 / 32** |
| DRAM throughput | 47.81 GB/s |
| No-eligible-warp cycles | 97.99% |
| Eligible warps per scheduler | 0.05 |
| Warp cycles per issued instruction | 551.48 |
| Achieved occupancy | 95.43% |
| Dynamic instructions | 77,612,532 |

The stall summaries attributed approximately 60.46% of sampled stalls to the LG instruction queue and 39.71% to L1TEX scoreboard dependencies.

### Conclusion

High occupancy did not imply useful throughput. Warps were resident but usually ineligible because each lane followed a separate, poorly coalesced load stream. The profiler evidence supported changing the work mapping.

## M3 — Block per Vector Pair

### Observation

In `cuda-naive`, adjacent warp lanes accessed the same dimension in different database rows. At `D=768`, those threads were about 3,072 bytes apart.

### Hypothesis

Assigning one block to each vector pair and distributing dimensions across threads would make neighboring lanes access neighboring FP32 elements and enable coalesced loads.

### Change

M3 introduced `cuda-block`: one 256-thread block per pair, lane-strided dimension work, and an explicit shared-memory reduction.

### Measurement

The paired `D=768` Nsight Compute run showed:

| Metric | `cuda-naive` | `cuda-block` |
|---|---:|---:|
| Global-load useful bytes per sector | 4 / 32 | **32 / 32** |
| DRAM throughput | 51.41 GB/s | 125.76 GB/s |
| No-eligible-warp cycles | 98.67% | 41.01% |
| Eligible warps per scheduler | 0.03 | 1.31 |
| Warp cycles per issued instruction | 860.95 | 19.02 |
| Achieved occupancy | 95.46% | 93.93% |

In the primary M4 benchmark, block kernel latency at `D=768` was 7.879 ms versus 64.447 ms for naive, an 8.180x speedup.

### Conclusion

The access redesign corrected the sector-utilization problem and substantially reduced memory-dependency pressure. The full-block reduction was effective for high dimension, but its barrier and instruction overhead penalized smaller dimensions.

## M4 — Warp per Vector Pair

### Observation

`cuda-block` fixed coalescing, but it assigned 256 threads and a full shared-memory reduction to every pair. At `D=128`, the block kernel was slower than the naive kernel in the dimension-sweep run.

### Hypothesis

A single warp could preserve coalesced dimension access while reducing synchronization, reduction, and instruction overhead.

### Change

M4 added `cuda-warp`: one warp per vector pair, lane-strided accumulation, and register-level `__shfl_down_sync` reduction. Eight independent pairs share each 256-thread block.

### Measurement

The M4 dimension sweep used `N=100,000`, `Q=4`, `K=10`, seed 42, two warm-ups, and five measured iterations:

| Dimension | `cuda-naive` (ms) | `cuda-block` (ms) | `cuda-warp` (ms) | Block speedup | Warp speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 3.388 | 5.712 | 1.123 | 0.593x | 3.017x |
| 256 | 19.149 | 5.559 | 2.351 | 3.445x | 8.145x |
| 768 | 63.601 | 7.975 | 6.985 | 7.975x | 9.105x |

The separate M4 primary run at `D=768` measured 64.447 ms for naive, 7.879 ms for block, and 6.918 ms for warp, yielding the headline **9.316x** warp-kernel speedup over naive.

Nsight Compute explains the workload-dependent behavior:

| Metric | D=768 block | D=768 warp | D=128 block | D=128 warp |
|---|---:|---:|---:|---:|
| Global-load useful bytes/sector | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 |
| DRAM throughput (GB/s) | 125.00 | 177.48 | 31.16 | 185.12 |
| No-eligible-warp cycles | 41.00% | 73.72% | 44.07% | 38.85% |
| Eligible warps/scheduler | 1.31 | 0.39 | 1.18 | 1.35 |
| Warp cycles/issued instruction | 19.01 | 40.98 | 19.45 | 15.91 |
| Dynamic instructions | 632.8 M | 202.4 M | 444.0 M | 80.4 M |

At `D=128`, the block kernel's leading sampled stall was the CTA barrier at 6.6 cycles per instruction, or 33.9% of the reported stall distribution. The warp kernel required no block-wide reduction barrier.

### Conclusion

Warp-per-pair is the best tested general-purpose kernel in this repository. It retains the coalescing benefit of M3 and substantially lowers execution overhead, especially for smaller dimensions. Block-per-pair remains instructive and competitive at high dimension, so it is preserved as a separate backend.

## M5 — GPU-Resident Database

### Observation

After the warp kernel became much faster, the stateless backend still allocated and uploaded the full database for every search. M2 had already shown database H2D as the second-largest stage; later repeated-query measurements made that system cost more visible.

### Hypothesis

Keeping database vectors on the GPU across query batches would remove redundant database allocation and H2D transfer from warm searches without changing the exact-search kernel.

### Change

M5 added the `cuda-warp-resident` lifecycle:

1. `prepare_database()` allocates and uploads the database once.
2. Repeated `search_prepared()` calls upload only queries, launch the existing warp kernel, copy scores back, and run CPU Top-K.
3. Reload and clear operations explicitly replace or release resident state.

### Measurement

#### Cold start

For the dedicated `N=100,000`, `D=768`, `Q=4`, `K=10` cold-start run:

| Mode | Measurement | Latency (ms) |
|---|---|---:|
| Stateless warp | Complete search | 526.642 |
| Resident | Database preparation | 660.766 |
| Resident | First query after preparation | 15.873 |
| Resident | Cold total | 676.639 |

Resident cold total was slower in this run. GPU residency is therefore not presented as a one-shot latency optimization.

#### Repeated warm queries

In the separate M5 primary repeated-query run:

| Stage | Stateless warp (ms) | Resident warm query (ms) |
|---|---:|---:|
| Database H2D | 369.096 | 0.000 |
| Kernel | 7.459 | 7.038 |
| Score D2H | 2.321 | 2.136 |
| CPU Top-K | 2.415 | 2.452 |
| End-to-end | 395.502 | 13.287 |

The measured warm-query ratio was approximately 29.76x. This is deliberately not the project's headline speedup: the stateless database upload in that run was much slower than the earlier M2 upload, so the ratio is environment-sensitive and applies only to repeated prepared searches.

The M5 amortization sweep reported the following total-work ratios:

| Query batches after one preparation | Resident speedup vs repeated stateless calls |
|---:|---:|
| 1 | 0.577x |
| 2 | 1.130x |
| 5 | 2.665x |
| 10 | 4.899x |
| 20 | 8.426x |
| 100 | 19.760x |

Nsight Systems API-count summaries for three searches also showed the intended lifecycle change:

| CUDA API category | Stateless | Resident |
|---|---:|---:|
| `cudaMalloc` calls | 9 | 7 |
| `cudaMemcpy` calls | 9 | 7 |
| `cudaFree` calls | 9 | 7 |
| Kernel launches | 3 | 3 |

The WSL2 Nsight Systems runs did not expose every requested GPU memory/kernel summary, so conclusions are limited to the counters and timings that were actually collected.

### Conclusion

M5 shifted database transfer from a per-search cost to an explicit preparation cost. It did not optimize the kernel, score download, or CPU Top-K. Its value depends on database reuse: cold one-shot work can regress, while sufficiently repeated query batches amortize preparation.

## Reconciling Separate Runs

The project intentionally retains raw outputs instead of forcing results into a single synthetic table:

- M2 stage timing, M3/M4 comparisons, and M5 lifecycle experiments were separate executions.
- Nominally identical `D=768` kernel values differ slightly across runs; this is normal measurement variance.
- The M5 stateless H2D value is an observed outlier relative to M2. It is retained and explicitly caveated rather than rewritten.
- Kernel-only, stateless end-to-end, resident preparation, and resident warm-query timings describe different scopes and must not be mixed without labels.

## Reproducibility Artifacts

- [`benchmarks/results/README.md`](../benchmarks/results/README.md) catalogs the retained CSV evidence.
- [`scripts/run_m2_benchmarks.py`](../scripts/run_m2_benchmarks.py), [`run_m3_benchmarks.py`](../scripts/run_m3_benchmarks.py), [`run_m4_benchmarks.py`](../scripts/run_m4_benchmarks.py), and [`run_m5_benchmarks.py`](../scripts/run_m5_benchmarks.py) reproduce the milestone benchmark suites.
- [`profiling/README.md`](../profiling/README.md) documents the retained summaries and profiler commands.
- [`scripts/profile_nsys.sh`](../scripts/profile_nsys.sh), [`profile_ncu.sh`](../scripts/profile_ncu.sh), and [`profile_m5_nsys.sh`](../scripts/profile_m5_nsys.sh) reproduce the profiler collection.
- [`scripts/plot_final_results.py`](../scripts/plot_final_results.py) regenerates the final figures directly from the M4 dimension CSV.

## Final Conclusion

The measured project story is:

1. Stage timing showed that the naive kernel was the largest end-to-end component.
2. Nsight Compute traced its cost to badly coalesced row-major loads and memory-dependency stalls.
3. Block-per-pair restored 32/32 useful bytes per sector and produced a large `D=768` speedup.
4. Warp-per-pair retained coalescing while lowering reduction overhead, reaching approximately 9.3x kernel speedup over naive in the primary `D=768` run.
5. With kernel cost reduced, repeated database transfer became a system bottleneck; GPU residency removed that recurring work for prepared searches.

M5 is the last implemented performance milestone. M5.5 finalizes documentation, evidence, visualization, and reproducibility without introducing a new optimization.
