# Final Architecture

## Purpose and invariants

The project implements exact similarity search over normalized FP32 vectors.
For a row-major database with `N` vectors of dimension `D` and `Q`
queries, every backend computes:

```text
scores[q, n] = dot(query[q], database[n])
```

The shared CPU Top-K stage returns the highest `K` scores for each query.
Ties are deterministic: lower database indices rank first. The serial CPU
backend is the correctness oracle for every CUDA implementation.

The architecture intentionally preserves the optimization stages as selectable
backends. This keeps comparisons controlled and makes the performance story
inspectable; the fastest design does not overwrite the baseline.

## Component map

```text
                         src/main.cpp
                  CLI parsing and orchestration
                              |
              +---------------+---------------+
              |                               |
              v                               v
     deterministic dataset            backend factory
      generation/normalization        create_backend(name)
              |                               |
              +---------------+---------------+
                              v
                         SearchRequest
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
      CpuSearchBackend                  CUDA backends
      serial dot products       naive / block / warp / resident
             |                                 |
             |                         device score array
             |                                 |
             +----------------+----------------+
                              v
                      shared CPU Top-K
                              |
                              v
                         SearchResult
                              |
                   benchmark/timing output
```

The executable, tests, and benchmark runners all use the same public request
and result types from `include/vector_search.hpp`.

## Public data model

### `SearchRequest`

A stateless search request contains host pointers to the row-major database and
queries plus `N`, `D`, `Q`, and `K`. CPU and stateless CUDA backends
require both pointers. A prepared resident backend accepts either the original
host database pointer for an identity check or a null database pointer.

### `SearchResult`

Results are stored query-major:

```text
query 0: rank 0 ... rank K-1
query 1: rank 0 ... rank K-1
...
```

Each `SearchHit` contains the database index and FP32 similarity score.
`SearchResult::at(query, rank)` bounds-checks access.

### `BackendTiming`

Backends expose timing from the most recent search. The structure separates:

- kernel-only CUDA Event latency;
- stateless allocation, database/query H2D, score D2H, CPU Top-K, cleanup, and
  complete E2E timing;
- resident database preparation; and
- resident per-query allocation, query H2D, kernel, score D2H, CPU Top-K,
  cleanup, and query E2E timing.

Detailed stage instrumentation is opt-in because extra events and
synchronization perturb small workloads.

## Backend abstraction

`SearchBackend` defines:

```cpp
virtual std::string name() const = 0;
virtual SearchResult search(const SearchRequest&) const = 0;
virtual BackendTiming last_timing() const;
virtual void set_timing_breakdown_enabled(bool);
```

`create_backend(name)` selects:

| Name | Implementation | State model |
| --- | --- | --- |
| `cpu` | `src/cpu/cpu_search.cpp` | Stateless |
| `cuda-naive` | `src/cuda/naive_search.cu` | Stateless |
| `cuda-block` | `src/cuda/block_search.cu` | Stateless |
| `cuda-warp` | `src/cuda/warp_search.cu` | Stateless |
| `cuda-warp-resident` | `src/cuda/resident_search.cu` | Explicit resident database |

CUDA names are rejected with a clear error in CPU-only builds.

`ResidentSearchBackend` extends the base interface with:

```cpp
prepare_database(DatabaseRequest)
reload_database(DatabaseRequest)
clear_database()
database_is_prepared()
database_info()
```

The separate interface makes database lifetime explicit without weakening the
stateless contract.

## Backend designs

### CPU reference

```text
for each query
    for each database vector
        serial FP32 dot product across D
    deterministic CPU Top-K
```

The implementation is deliberately straightforward. It is a correctness
reference, not a parallel CPU optimization.

### `cuda-naive`: one thread per pair

```text
grid covers Q * N pairs
thread i -> (query i/N, vector i%N)
thread   -> serial loop over D
thread   -> one score
```

The launch uses 256 threads per block. Because adjacent threads own adjacent
rows, at a fixed dimension their database loads are `D * sizeof(float)`
bytes apart. This frozen M1 implementation is the profiling baseline.

### `cuda-block`: one block per pair

```text
grid block                 -> one pair
thread t                   -> dimensions t, t+256, t+512, ...
256 partial sums           -> shared-memory array
synchronized tree reduce  -> one score
```

Adjacent threads access adjacent dimensions, producing coalesced loads. The
fixed 256-thread block and full reduction are intentionally preserved as the
M3 experimental design.

### `cuda-warp`: one warp per pair

```text
256-thread block = 8 warps
global warp ID   -> one pair
lane l           -> dimensions l, l+32, l+64, ...
shuffle reduction at 16, 8, 4, 2, 1
lane 0           -> one score
```

A ballot-derived mask is computed before a tail warp may return. Valid pair
warps therefore use one consistent mask for all `__shfl_down_sync` steps.
This backend uses no explicit reduction shared memory and no block-wide
barrier.

### `cuda-warp-resident`: M4 kernel, persistent database

The resident backend duplicates the M4 warp kernel intentionally so the M5
controlled variable is database lifetime, not arithmetic or launch mapping.
Query and score buffers remain request-local.

## Stateless CUDA data flow

All three stateless CUDA backends use the same host/device pipeline:

```text
host database --cudaMalloc/cudaMemcpy H2D--+
                                                |
host queries  --cudaMalloc/cudaMemcpy H2D--+   |
                                            v   v
                                  similarity kernel
                                            |
                                      device scores
                                            |
                               cudaMemcpy D2H all Q*N scores
                                            |
                                      shared CPU Top-K
                                            |
                               destroy events / cudaFree buffers
```

The only intended difference is the kernel's mapping and reduction.

Each backend:

1. validates pointers, dimensions, query count, and `K`;
2. checks products and byte sizes for `size_t` overflow;
3. validates the required grid against the device limit;
4. checks every CUDA allocation, copy, event, kernel launch, synchronization,
   and explicit release;
5. copies the full score array to the host; and
6. calls the shared deterministic CPU Top-K implementation.

Device buffers and events use small RAII wrappers. Destructors report cleanup
errors; explicit releases use the normal throwing error checks.

## Resident database lifecycle

```text
construct
   |
   v
unprepared
   |
   | prepare_database(database, N, D)
   v
prepared ------------------------------------+
   |                                         |
   | search(query batch)                     | reload_database(...)
   |   allocate query/score buffers          | releases old database
   |   query H2D                             | allocates/uploads new database
   |   unchanged M4 warp kernel              |
   |   score D2H + CPU Top-K                 |
   |   release query/score buffers           |
   +---------------- repeat -----------------+
   |
   | clear_database() or destruction
   v
unprepared
```

### Preparation semantics

`prepare_database` validates the database, reads free-device-memory metadata,
allocates one device buffer, performs one synchronous database H2D copy, and
records its shape and original host pointer. Repeating preparation with the
same pointer, `N`, and `D` is idempotent.

`reload_database` forces a release and re-upload, including when the host
pointer is unchanged. A failed preparation leaves the backend unprepared.

### Search semantics

Search before preparation throws. A query must match the prepared `N` and
`D`. A non-null database pointer must equal the prepared host pointer,
preventing accidental use of a different host database while the old device
database remains active.

`clear_database` releases the persistent device allocation and resets
identity, shape, memory metadata, and last timing.

## Timing and benchmark subsystem

`src/common/benchmark.cpp` provides two measurement paths.

### Ordinary benchmark

`benchmark_search` performs configured warm-ups followed by measured calls:

```text
host steady_clock around SearchBackend::search
CUDA Event timing reported by the backend for its kernel
optional stage means from BackendTiming
```

Dataset generation occurs before the benchmark and is excluded. Min, mean,
max, queries/second, and a result checksum are reported. The checksum keeps
the result observable during benchmarks; correctness is established by the
test suite rather than checksum equality alone.

### Repeated-batch benchmark

`benchmark_repeated_search` accepts an ordered request sequence, runs warm-up
batches, and records every measured batch's host latency and backend timing.
The CLI generates different deterministic query contents while reusing one
database. This is the M5 path for comparing stateless searches with resident
warm queries.

### Timing boundaries

- CUDA Events measure synchronous copies and kernels.
- `std::chrono::steady_clock` measures host allocation calls, CPU Top-K,
  cleanup, and E2E scope.
- A diagnostic component sum need not exactly equal E2E because device
  metadata calls, host container work, event creation, and timing boundaries
  are not all separate rows.
- Nsight replay duration is profiler evidence, not the benchmark headline.
- Resident cold cost is preparation plus first-query E2E; warm query latency
  excludes preparation.

## Correctness and test architecture

The CPU result is compared with each CUDA backend using tolerance-based score
checks and exact result indices. Coverage includes:

- deterministic known vectors;
- normalized generated inputs;
- `Q=1` and multiple queries;
- `K=1` and `K>1`;
- dimensions below, at, above, and not divisible by warp/block boundaries;
- partially populated final warp blocks;
- stage timing enabled;
- resident search before preparation;
- preparation metadata;
- repeated searches against one database;
- forced reload;
- incompatible request rejection; and
- search after clear.

CMake registers one CPU test executable in all builds and a second CUDA test
executable when `ENABLE_CUDA=ON`.

## Build architecture

`ENABLE_CUDA=OFF` is the default. The core library always contains dataset,
benchmark, CPU search, and CPU Top-K sources. Enabling CUDA:

1. enables the CMake CUDA language;
2. requires the CUDA Toolkit;
3. adds all four `.cu` backend sources;
4. defines `VECTOR_SEARCH_ENABLE_CUDA=1`; and
5. links `CUDA::cudart`.

Both C++ and CUDA use C++17. Release builds request `-O3` on GCC/NVCC or
`/O2` on MSVC. CUDA separable compilation is enabled.

## Repository structure

```text
.
├── CMakeLists.txt
├── README.md
├── include/                  public requests, results, timing, backend APIs
├── src/
│   ├── common/               dataset and benchmark implementation
│   ├── cpu/                  serial similarity and deterministic Top-K
│   └── cuda/                 naive, block, warp, and resident backends
├── tests/                    CPU, cross-backend, and resident lifecycle tests
├── scripts/                  benchmark, plotting, Nsight Systems/Compute tools
├── benchmarks/results/       curated raw CSV evidence and catalog
├── profiling/                compact profiler exports and artifact policy
└── docs/                     architecture, experiment log, report, interview notes
```

## Deliberate boundaries

The implemented architecture is synchronous, single-GPU, exact FP32 search.
It retains full score D2H and CPU Top-K. There is no automatic backend
selection, GPU Top-K, persistent query buffer, stream pipeline, pinned memory,
asynchronous copy, FP16/Tensor Core path, external CUDA primitive/library,
ANN index, distributed service, API server, or frontend.

Those items are potential future experiments, not hidden or partially
implemented parts of the final M0–M5 architecture. Historical reasoning and
milestone measurements belong in `docs/optimization_log.md` and
`docs/performance_report.md`; this document describes the repository as it
exists after M5.5.
