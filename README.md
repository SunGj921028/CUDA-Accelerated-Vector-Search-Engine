# CUDA-Accelerated Vector Search Engine

This repository is an educational exact vector-search engine built in stages.
M0 is the deterministic CPU reference implementation. M1 adds the intentionally
naive `cuda-naive` backend as a measurable CUDA baseline; later milestones may
add separate backends without changing the backend-neutral search contract.

The project computes exact dot-product similarity for normalized FP32 vectors
and returns deterministic Top-K results. Equal scores are ordered by vector
index. It does not include approximate nearest-neighbor search, a vector
database, RAG, LLM integration, or third-party search libraries.

## Current M1 environment

The tested CUDA environment is:

- Ubuntu 22.04.4 LTS under WSL2;
- NVIDIA GeForce RTX 3050 Laptop GPU class hardware (the validation host
  reports `NVIDIA GeForce RTX 3050 Ti Laptop GPU`) with 4 GB VRAM;
- NVIDIA driver 576.88;
- CUDA Toolkit 12.9, NVCC 12.9.86;
- GCC/G++ 11.4.0; and
- CMake 3.22.1.

CUDA is expected to be installed and functional already. Do not modify NVIDIA
drivers or system CUDA libraries as part of this project.

## Build

### CPU-only build

CPU-only builds remain supported and do not require CUDA:

```bash
cmake -S . -B build-cpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=OFF
cmake --build build-cpu --parallel
```

`ENABLE_CUDA` defaults to `OFF`, so existing CPU-only users can omit it.

### CUDA-enabled build

Configure from the WSL2 repository checkout with the CUDA language enabled:

```bash
cmake -S . -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=ON
cmake --build build-cuda --parallel
```

If `nvcc` is installed outside `PATH`, provide its existing path without
changing the system installation, for example:

```bash
cmake -S . -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
```

The M1 CMake option isolates CUDA sources and links the CUDA runtime only in
the CUDA-enabled configuration. No Windows/MSVC-specific CUDA configuration is
required.

## Tests

Run the CPU-only test suite:

```bash
ctest --test-dir build-cpu --output-on-failure
```

Run both the CPU regression tests and CUDA correctness tests from the CUDA
build:

```bash
ctest --test-dir build-cuda --output-on-failure
```

The CUDA tests compare `cuda-naive` against the CPU backend with a floating-
point tolerance. They cover normalized inputs, multiple queries, an irregular
dimension (`D = 3`), a database size smaller than a 256-thread block, `K = 1`,
and `K > 1`.

## Selecting a backend

The CLI uses the same backend-neutral request for CPU and CUDA:

```bash
./build-cpu/vector_search \
    --backend cpu \
    --vectors 1000 --dim 32 --queries 2 --topk 5 --seed 7

./build-cuda/vector_search \
    --backend cuda-naive \
    --vectors 1000 --dim 32 --queries 2 --topk 5 --seed 7
```

The normal search mode prints the Top-K results for query 0. Use `--help` for
all command-line options.

## M1 naive CUDA design

For `Q` queries and `N` database vectors, M1 launches one CUDA thread for each
of the `Q * N` query/database-vector pairs. Each thread:

1. maps its global index to `query_idx = idx / N` and
   `vector_idx = idx % N`;
2. sequentially loops over all `D` dimensions;
3. computes one FP32 dot product; and
4. writes one score to the flat, query-major device score array.

The kernel uses a fixed `256` threads per block. The host pipeline is

```text
host database  --H2D--> device database
host queries   --H2D--> device queries
                          |
                          v
                 naive similarity kernel
                          |
                 device similarity scores
                          |
                          +--D2H--> host scores --> existing CPU Top-K
```

M1 intentionally copies the complete score array back to the CPU and does not
implement GPU Top-K, shared memory, reductions, streams, pinned memory,
vectorized loads, block-size tuning, or any other M2/M3 optimization.

## Benchmarking

Benchmark mode performs warm-up searches followed by multiple measured
iterations. Dataset generation is excluded. For every backend, the regular
`latency_ms_*` values measure the complete `SearchBackend::search()` call.
For `cuda-naive`, that end-to-end scope includes per-search device allocation,
H2D copies, kernel execution, D2H score copy, CPU Top-K, and device cleanup.

CUDA additionally reports:

- `kernel_latency_ms_*`: CUDA Event timing around only the similarity kernel;
- `end_to_end_latency_ms_average`: the same complete backend timing as
  `latency_ms_average`; and
- GPU model and CUDA Toolkit version.

Use identical arguments and seed for CPU and CUDA comparisons. The required M1
workloads use four queries, `K = 10`, seed `42`, two warm-ups, and five measured
iterations:

```bash
./build-cuda/vector_search --backend cpu \
    --vectors 10000 --dim 128 --queries 4 --topk 10 --seed 42 \
    --benchmark --warmup 2 --iterations 5
./build-cuda/vector_search --backend cuda-naive \
    --vectors 10000 --dim 128 --queries 4 --topk 10 --seed 42 \
    --benchmark --warmup 2 --iterations 5

./build-cuda/vector_search --backend cpu \
    --vectors 100000 --dim 128 --queries 4 --topk 10 --seed 42 \
    --benchmark --warmup 2 --iterations 5
./build-cuda/vector_search --backend cuda-naive \
    --vectors 100000 --dim 128 --queries 4 --topk 10 --seed 42 \
    --benchmark --warmup 2 --iterations 5

./build-cuda/vector_search --backend cpu \
    --vectors 100000 --dim 768 --queries 4 --topk 10 --seed 42 \
    --benchmark --warmup 2 --iterations 5
./build-cuda/vector_search --backend cuda-naive \
    --vectors 100000 --dim 768 --queries 4 --topk 10 --seed 42 \
    --benchmark --warmup 2 --iterations 5
```

M1 does not chunk or stream workloads. CUDA allocation failures are surfaced as
errors. In particular, do not use `1,000,000 x 768` as a normal M1 benchmark
on a 4 GB GPU.

## Known CMake warning

On the validation WSL2 host, CMake prints:

```text
/usr/bin/cmake: /usr/local/lib/libcurl.so.4:
no version information available
```

This is a pre-existing environment warning caused by CMake resolving the
`/usr/local/lib` libcurl. It is separate from CUDA. If configure, build, and
tests complete, it is not treated as an M1 failure. Do not delete, replace, or
modify `/usr/local/lib/libcurl*`.
