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
