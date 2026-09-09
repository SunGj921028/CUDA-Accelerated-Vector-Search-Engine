# CUDA-Accelerated Vector Search Engine

This repository is an educational exact vector-search engine built in stages.
M0 is the deterministic CPU reference implementation. M1 adds the intentionally
naive cuda-naive backend as a measurable CUDA baseline. M3 and M4 add
separate block-per-vector and warp-per-vector kernels, and M5 adds an explicit
GPU-resident database lifecycle without changing the stateless backend contract.

The project computes exact dot-product similarity for normalized FP32 vectors
and returns deterministic Top-K results. Equal scores are ordered by vector
index. It does not include approximate nearest-neighbor search, a
general-purpose vector database, RAG, LLM integration, or third-party search
libraries.

## Current M5 environment

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

## M2 profiling and latency decomposition

Detailed stage timing is opt-in so the normal `cuda-naive` benchmark remains
the M1 baseline measurement. Use `--timing-breakdown` with a benchmark to
report average timings for allocation, H2D database/query copies, the
similarity kernel, D2H score copy, CPU Top-K, cleanup, and the measured
end-to-end backend call:

```bash
./build-cuda/vector_search --backend cuda-naive \
    --vectors 100000 --dim 768 --queries 4 --topk 10 --seed 42 \
    --benchmark --timing-breakdown --warmup 2 --iterations 5
```

CUDA Events measure the synchronous H2D, kernel, and D2H stages. Host
`steady_clock` measures allocation, CPU Top-K, cleanup, and the fully
synchronized backend call. The stage sum is diagnostic and is not forced to
equal the end-to-end value; event creation, host container allocation, device
metadata queries, and measurement boundaries can remain outside the listed
components.

The standard-library benchmark runner writes small CSV files for the required
crossover sweep, dimension sweep, and representative stage breakdown:

```bash
python3 scripts/run_m2_benchmarks.py --binary build-cuda/vector_search
```

The Nsight Systems and Nsight Compute command wrappers are:

```bash
scripts/profile_nsys.sh build-cuda/vector_search
scripts/profile_ncu.sh build-cuda/vector_search
```

Both default to temporary output directories. The Nsight Compute wrapper
collects focused SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, LaunchStats,
and WarpStateStats sections for one `D=768` naive-kernel launch. Under WSL2,
GPU performance-counter access may need to be enabled on the Windows host;
the wrapper reports `ERR_NVGPUCTRPERM` without changing security settings.

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


## M5 GPU-resident database

The cuda-warp-resident backend makes the database lifecycle explicit. Prepare
the database once, issue multiple searches, and then clear or destroy the
backend:

~~~text
prepare_database -> search(batch 1) -> search(batch 2) -> ... -> clear_database
~~~

The existing cuda-warp backend remains stateless and still performs database
allocation and H2D upload on every search. The resident backend does not alter
the M4 warp kernel, query/score buffer policy, or CPU Top-K path.

For the repeated-query CLI experiment, use the same database with sequential
query batches:

~~~bash
./build-m5-cuda-wsl/vector_search \
    --backend cuda-warp-resident \
    --vectors 100000 --dim 768 --queries 4 --topk 10 --seed 42 \
    --timing-breakdown --warmup 2 --repeat-batches 100
~~~

The output separates database preparation fields from warm query fields. Use
the M5 runner to compare stateless and resident backends, including cold
start, warm means/medians, amortization at 1/2/5/10/20/100 batches, D=128,
D=256, D=768, and the optional 250K workload:

~~~bash
python3 scripts/run_m5_benchmarks.py \
    --binary build-m5-cuda-wsl/vector_search \
    --output-dir benchmarks/results \
    --warmup 2 --iterations 100 --include-250k
~~~

The representative system-level trace is captured with:

~~~bash
bash scripts/profile_m5_nsys.sh \
    build-m5-cuda-wsl/vector_search profiling/nsys/m5
~~~

M5 is intended for repeated searches against an unchanged database. Its cold
path includes preparation and can be slower than a one-shot stateless search;
warm latency must not be described as startup latency.
