# CUDA Backends

The repository preserves each measured CUDA design as a separate backend so
the optimization progression remains inspectable and reproducible. All
backends compute exact FP32 dot products and return results through the shared
deterministic CPU Top-K implementation.

| Backend | Source | Work mapping | Reduction |
|---|---|---|---|
| `cuda-naive` | `naive_search.cu` | One thread per vector pair | Serial in one thread |
| `cuda-block` | `block_search.cu` | One 256-thread block per pair | Shared memory + block barriers |
| `cuda-warp` | `warp_search.cu` | One 32-lane warp per pair; eight pairs per block | `__shfl_down_sync` |
| `cuda-warp-resident` | `resident_search.cu` | Same warp kernel as `cuda-warp` | Same warp reduction |

## Stateless Backends

`cuda-naive`, `cuda-block`, and `cuda-warp` use the same host pipeline:

```text
allocate device buffers
  -> upload database and queries
  -> execute the selected kernel
  -> download the complete score matrix
  -> run CPU Top-K
  -> release device buffers
```

Keeping this pipeline constant isolates the effect of the M1, M3, and M4
kernel mappings. Every CUDA API operation and kernel launch is checked.

## Resident Backend

M5 adds an explicit `ResidentSearchBackend` lifecycle:

```text
prepare_database()
  -> search_prepared() repeatedly
  -> reload_database() or clear_database()
```

Preparation owns the database device allocation and one synchronous H2D
upload. Each prepared search still uploads queries, launches the exact M4 warp
kernel, downloads scores, and runs CPU Top-K. Query and score buffers remain
per-search allocations. Preparation timing is reported separately from warm
query timing, and CUDA resources are held through RAII.

Resident mode is a data-lifecycle change, not a new kernel. It is intended for
repeated query batches and is not claimed to improve one-shot cold latency.

See [`docs/architecture.md`](../../docs/architecture.md) for component and
lifecycle details and [`docs/performance_report.md`](../../docs/performance_report.md)
for the measured progression.
