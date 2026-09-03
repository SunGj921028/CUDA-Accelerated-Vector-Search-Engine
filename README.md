# CUDA-Accelerated Vector Search Engine

This repository is an educational exact vector-search engine built in stages.
Milestone M0 provides the reproducible CPU reference implementation; CUDA
backends will be added in later milestones without changing the CLI or
benchmark contract.

## M0 scope

The CPU backend:

- generates deterministic normalized FP32 database and query vectors;
- computes exact dot-product similarities; and
- selects deterministic Top-K results, breaking equal-score ties by index.

The implementation intentionally has no CUDA kernels, approximate nearest
neighbor index, vector database, RAG layer, or third-party search library.

## Build

Requirements:

- CMake 3.18 or newer;
- a C++17 compiler.

Configure a Release build from the repository root:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The CMake project does not enable the CUDA language in M0. Release builds use
the compiler's optimized configuration (`-O3` for non-MSVC toolchains and
`/O2` for MSVC).

## Tests

Run all registered CPU correctness tests:

```powershell
ctest --test-dir build --build-config Release --output-on-failure
```

The tests cover vector normalization, FP32 similarity, Top-K ordering and
tie-breaking, deterministic seeded data generation, and the multi-query CPU
backend contract.

## Run a small CPU search

```powershell
.\build\vector_search.exe --backend cpu --vectors 1000 --dim 32 --queries 2 --topk 5 --seed 7
```

The normal search mode prints the Top-K results for query 0. The command-line
options are also available through `--help`:

```powershell
.\build\vector_search.exe --help
```

## CPU benchmark mode

Benchmark mode warms up the backend, then reports minimum, average, and
maximum search-only latency over multiple iterations. Dataset generation is
excluded from the timed region.

```powershell
.\build\vector_search.exe `
    --backend cpu `
    --vectors 10000 `
    --dim 128 `
    --queries 8 `
    --topk 10 `
    --seed 42 `
    --benchmark `
    --warmup 2 `
    --iterations 5
```

Benchmark output records the backend, CPU logical-processor count, compiler,
GPU/CUDA availability, workload dimensions, seed, timing scope, iteration
counts, latency, throughput, and a result checksum.

## M0 design for M1

The CLI constructs a backend-neutral `SearchRequest` and obtains a
`SearchBackend` through `create_backend`. Backends return a flat,
query-major `SearchResult`, so the benchmark can call CPU or future CUDA
implementations through the same interface. CPU scoring and Top-K are separate
functions because they are the correctness reference for later CUDA stages.

When M1 is implemented, the CUDA backend should be added as another
`SearchBackend` implementation and selected by `--backend`; the CPU backend
must remain available for tolerance-based correctness comparisons.
