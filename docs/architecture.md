# CUDA-Accelerated Vector Search Engine

## 1. Project Overview

### Objective

Design and implement a high-performance exact vector similarity search engine using C++ and CUDA C++, then iteratively profile and optimize the GPU implementation.

The project focuses on understanding and improving GPU performance rather than building a complete RAG application.

Given:

* A database containing `N` high-dimensional vectors
* One or more query vectors
* Vector dimension `D`

the system computes similarity scores between each query and database vectors and returns the Top-K most similar vectors.

### Primary Goals

* Implement a correct CPU reference implementation in C++
* Parallelize vector similarity computation using CUDA C++
* Understand CUDA thread/block/grid execution
* Analyze GPU memory access patterns
* Optimize kernel execution and memory throughput
* Implement or integrate GPU-side Top-K selection
* Profile workloads using NVIDIA Nsight Systems and Nsight Compute
* Benchmark CPU and GPU implementations systematically
* Produce a reproducible performance report

### Non-Goals

The first version will NOT include:

* RAG
* LLM APIs
* Embedding models
* Vector databases
* Web interfaces
* Distributed or multi-GPU execution
* Approximate nearest-neighbor algorithms

These may be added later, but they are intentionally excluded from the core project.

---

# 2. Core Problem

For normalized vectors:

```
database = N × D
query    = 1 × D
```

Cosine similarity becomes:

```
similarity(q, x) = dot(q, x)
```

Therefore, the core computational problem becomes:

```
scores[i] = dot(query, database[i])
```

followed by:

```
TopK(scores)
```

Example workloads:

```
N = 10,000
    100,000
    500,000
    1,000,000

D = 128
    256
    768
```

The system should automatically scale the workload according to available GPU memory.

---

# 3. High-Level Architecture

```
                     ┌─────────────────────┐
                     │       CLI / App     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │    Dataset Loader   │
                     │  / Data Generator   │
                     └──────────┬──────────┘
                                │
               ┌────────────────┴────────────────┐
               │                                 │
               ▼                                 ▼
    ┌─────────────────────┐          ┌─────────────────────┐
    │    CPU Backend      │          │    CUDA Backend     │
    │                     │          │                     │
    │ Scalar C++          │          │ Naive Kernel        │
    │ OpenMP (optional)   │          │ Optimized Kernels   │
    └──────────┬──────────┘          └──────────┬──────────┘
               │                                │
               └──────────────┬─────────────────┘
                              ▼
                     ┌─────────────────────┐
                     │       Top-K         │
                     │     Selection       │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Verification Layer  │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Benchmark / Profile │
                     └─────────────────────┘
```

---

# 4. Repository Structure

cuda-vector-search/
│
├── CMakeLists.txt
├── README.md
├── AGENTS.md
│
├── include/
│   ├── vector_search.hpp
│   ├── dataset.hpp
│   ├── benchmark.hpp
│   └── cuda_utils.hpp
│
├── src/
│   ├── main.cpp
│   │
│   ├── common/
│   │   ├── dataset.cpp
│   │   └── benchmark.cpp
│   │
│   ├── cpu/
│   │   ├── cpu_search.cpp
│   │   └── cpu_topk.cpp
│   │
│   └── cuda/
│       ├── naive_search.cu
│       ├── block_search.cu
│       ├── optimized_search.cu
│       └── gpu_topk.cu
│
├── tests/
│   ├── test_cpu.cpp
│   ├── test_cuda.cu
│   └── test_correctness.cpp
│
├── benchmarks/
│   ├── run_benchmarks.py
│   ├── configs/
│   └── results/
│
├── scripts/
│   ├── profile_nsys.sh
│   └── profile_ncu.sh
│
├── docs/
│   ├── architecture.md
│   ├── benchmark_methodology.md
│   ├── optimization_log.md
│   └── performance_report.md
│
└── profiling/
├── nsight_systems/
└── nsight_compute/

---

# 5. Backend Interface

CPU and CUDA implementations should expose a similar interface.

Conceptually:

```
SearchResult search(
    const float* database,
    const float* queries,
    size_t num_vectors,
    size_t dimension,
    size_t num_queries,
    size_t k
);
```

This prevents benchmarking code from being tightly coupled to one implementation.

Different implementations can then be selected from the CLI:

```
./vector_search --backend cpu

./vector_search --backend cuda-naive

./vector_search --backend cuda-block

./vector_search --backend cuda-optimized
```

---

# 6. Development Milestones

## M0 — Infrastructure and CPU Reference

Implement:

* CMake build system
* deterministic synthetic data generator
* vector normalization
* CPU dot-product similarity
* CPU Top-K
* correctness tests
* benchmark infrastructure
* CLI

The CPU implementation is the correctness oracle for every subsequent CUDA version.

Requirements:

* deterministic random seed
* reproducible test cases
* configurable N, D, query count, K
* Release build using compiler optimization
* automated tests

Example:

```
./vector_search \
    --backend cpu \
    --vectors 100000 \
    --dim 768 \
    --queries 1 \
    --topk 10
```

---

## M1 — Naive CUDA Implementation

Implement the first CUDA kernel with intentionally simple parallelization.

Example strategy:

```
one CUDA thread → one database vector
```

Each thread computes:

```
score[i] = dot(query, database[i])
```

Learning objectives:

* CUDA kernel syntax
* cudaMalloc
* cudaMemcpy
* kernel launch
* thread/block/grid indexing
* cudaDeviceSynchronize
* CUDA error handling
* cudaEvent timing

Do NOT optimize prematurely.

The goal is:

```
Correctness first.
```

Then benchmark:

```
CPU vs Naive CUDA
```

---

## M2 — Parallel Reduction Kernel

Redesign the similarity computation.

Instead of:

```
one thread → one entire vector
```

use:

```
one block → one vector
multiple threads → dimensions within that vector
```

Example:

```
           Vector #42

thread 0  → dimensions 0, 256, 512...
thread 1  → dimensions 1, 257, 513...
thread 2  → dimensions 2, 258, 514...
                    ...
                   ↓
            partial results
                   ↓
          parallel reduction
                   ↓
            similarity score
```

Implement:

* block-level parallelism
* shared memory reduction
* synchronization
* coalesced global-memory access

Compare against M1.

Document WHY the performance changes.

---

## M3 — Memory Optimization

Profile the kernel before changing it.

Investigate:

* global memory throughput
* memory access efficiency
* memory coalescing
* shared memory usage
* occupancy
* registers
* block size
* warp efficiency

Experiment with:

* different block sizes
* vectorized memory access where appropriate
* query-vector reuse
* shared-memory caching
* reduced redundant global-memory access

Do not blindly apply optimizations.

Every optimization must follow:

```
hypothesis
    ↓
profiler evidence
    ↓
implementation
    ↓
benchmark
    ↓
accept / reject
```

Record experiments in:

```
docs/optimization_log.md
```

Example:

```
Optimization:
Change block size from 256 → 128

Hypothesis:
Higher occupancy may hide memory latency.

Result:
Kernel latency decreased from X ms → Y ms.

Conclusion:
Keep / reject change.
```

---

## M4 — Top-K Optimization

Initial implementation:

```
GPU similarity
    ↓
copy all scores GPU → CPU
    ↓
CPU Top-K
```

This is deliberately simple.

Profile the complete pipeline.

If score transfer becomes significant, move Top-K processing onto the GPU.

Possible progression:

```
CPU Top-K

    ↓

GPU library-based Top-K / sorting reference

    ↓

custom GPU Top-K or block-level selection
(optional advanced implementation)
```

The important engineering story is:

```
Profiling revealed unnecessary GPU→CPU transfer.

Therefore Top-K was moved closer to the computation.
```

This provides a stronger system-level optimization story than simply implementing another kernel.

---

## M5 — Batched Queries and Streams

Extend:

```
1 query
```

to:

```
Q queries
```

Example:

```
Q = 1
    8
    32
    128
```

Investigate:

* batch size
* GPU utilization
* host/device transfer overhead
* asynchronous memory transfer
* CUDA streams
* pinned host memory
* computation-transfer overlap

Example conceptual pipeline:

```
Batch 1    H2D → Compute → D2H
Batch 2          H2D → Compute → D2H
Batch 3                H2D → Compute → D2H
```

Measure whether asynchronous execution actually improves end-to-end throughput.

---

# 7. Correctness Strategy

Performance optimizations must never silently break correctness.

For every CUDA implementation:

```
CPU result
    vs
CUDA result
```

Validate:

* similarity values within floating-point tolerance
* identical or equivalent Top-K results
* multiple dimensions
* multiple dataset sizes
* edge cases

Suggested tests:

* N < block size
* D not divisible by block size
* K = 1
* K > 1
* duplicate similarity scores
* small deterministic vectors
* large random vectors

Use tolerance-based comparison rather than exact floating-point equality.

---

# 8. Benchmark Methodology

Measure two different quantities.

## Kernel Performance

CUDA kernel execution only.

Use CUDA events.

Measure:

* kernel latency
* throughput
* optionally effective bandwidth

## End-to-End Performance

Measure:

```
host setup
  +
H2D transfer
  +
GPU computation
  +
Top-K
  +
D2H transfer
```

This avoids presenting misleading speedups.

Benchmark configurations should include combinations such as:

```
N:
10K
100K
500K
1M

D:
128
256
768

Query batch:
1
32
128
```

For reference:

```
1,000,000 × 768 × FP32
```

requires roughly 3.1 GB purely for database-vector storage, so workloads must respect GPU memory limits.

Every benchmark record should contain:

* GPU model
* CPU model
* CUDA version
* compiler version
* build configuration
* N
* D
* query count
* K
* latency
* throughput
* speedup

Run warm-up iterations before collecting measurements.

Perform multiple measured iterations rather than reporting a single execution.

---

# 9. Profiling Strategy

## Nsight Systems

Use Nsight Systems to understand the complete application timeline.

Investigate:

* CPU activity
* CUDA API calls
* kernel launches
* H2D transfers
* D2H transfers
* synchronization
* GPU idle gaps

Primary question:

```
Where is the application spending its time?
```

## Nsight Compute

Use Nsight Compute after identifying important kernels.

Investigate:

* memory throughput
* memory access efficiency
* achieved occupancy
* warp behavior
* instruction utilization
* shared-memory behavior
* kernel bottlenecks

Primary question:

```
Why is this kernel taking this amount of time?
```

Store selected reports under:

```
profiling/
```

and summarize findings in:

```
docs/performance_report.md
```

---

# 10. Performance Experiment Log

Every optimization should be documented.

Example:

## Experiment 007 — Block Size

### Baseline

```
threads/block = 256
latency = ...
```

### Hypothesis

Reducing block size may improve occupancy because the current kernel uses a relatively high number of registers per block.

### Change

```
threads/block:
256 → 128
```

### Result

```
latency:
X → Y

occupancy:
A% → B%
```

### Decision

Keep / revert.

### Interpretation

Explain WHY the result occurred.

This log is an important project artifact because it demonstrates performance-engineering reasoning rather than only final code.

---

# 11. Minimum Resume-Ready Version

The project is considered resume-ready when all of the following exist:

* C++ CPU baseline
* CUDA C++ implementation
* at least two CUDA kernel designs
* correctness tests
* reproducible benchmark suite
* Nsight profiling
* at least two evidence-driven optimizations
* CPU vs GPU performance comparison
* documented hardware/software environment
* clear README
* performance report

GPU Top-K, CUDA streams, OpenMP, FP16, Tensor Cores, multi-GPU, and RAG integration are extensions rather than requirements.

---

# 12. Stretch Goals

After the core project is complete, possible extensions include:

### OpenMP CPU Backend

Compare:

```
serial C++
vs
OpenMP
vs
CUDA
```

### cuBLAS Baseline

Express batched similarity computation as matrix multiplication and compare custom CUDA kernels against cuBLAS.

This answers:

```
How close is the custom implementation to a highly optimized NVIDIA library?
```

### FP16

Evaluate:

```
FP32
vs
FP16
```

Measure:

* throughput
* memory usage
* numerical error

### Roofline Analysis

Determine whether kernels are:

* compute-bound
* memory-bound

### Approximate Search

Only after exact search is complete, investigate ANN approaches.

### RAG Integration

Only as a final optional demonstration:

```
embeddings
    ↓
custom CUDA vector search
    ↓
retrieved documents
    ↓
LLM
```

The RAG layer should remain a consumer of the vector-search engine rather than the core of this project.

---

# 13. Final Deliverables

The GitHub repository should ultimately contain:

1. Working C++/CUDA source code
2. Unit and correctness tests
3. CMake build configuration
4. Reproducible benchmark scripts
5. Profiling scripts
6. Nsight analysis
7. Optimization experiment log
8. Performance comparison tables
9. Architecture documentation
10. Performance-focused README

The README should tell the story:

```
Problem
  ↓
CPU baseline
  ↓
Naive CUDA
  ↓
Profiling
  ↓
Bottleneck
  ↓
Optimization
  ↓
Benchmark
  ↓
Conclusions
```

The goal is not simply:

```
"I implemented vector search with CUDA."
```

The goal is:

```
"I analyzed a compute-intensive workload, parallelized it on an NVIDIA GPU,
profiled the implementation, identified bottlenecks, and iteratively
optimized its performance."
```
