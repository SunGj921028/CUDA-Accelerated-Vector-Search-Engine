# Profiling artifact policy

The repository keeps profiler commands and compact, reviewable evidence in
version control. Large generated reports and profiler databases remain local
and are ignored by Git.

## Tracked evidence

- `ncu/*_details.csv`: Nsight Compute section/metric exports for the naive
  kernel.
- `ncu/m3/*.csv`: the M3 naive/block comparison, including direct sector and
  stall counters.
- `ncu/m4/*.csv`: the M4 block/warp profiles at `D=128` and `D=768`.
- `nsys/m5/*_summary.txt`: Nsight Systems CUDA API summaries showing the
  stateless and resident lifecycle call counts.

These files are the source for the concise profiler tables in
`docs/performance_report.md` and `docs/optimization_log.md`.

## Generated and ignored

The following formats are intentionally ignored because they are generated,
binary, comparatively large, or easy to reproduce:

```text
*.ncu-rep
*.nsys-rep
*.qdrep
*.sqlite
```

Existing local copies do not need to be deleted to keep the public repository
clean; `git status --ignored` can confirm their ignore rules. Generate new
reports in `/tmp` when only a transient inspection is needed.

## Reproduce

```bash
# Nsight Systems: application/API sequencing for the naive backend
bash scripts/profile_nsys.sh build-cuda/vector_search /tmp/vector-search-nsys

# Nsight Compute: one selected kernel launch
bash scripts/profile_ncu.sh \
    build-cuda/vector_search /tmp/vector-search-ncu 100000 768 cuda-naive

# Nsight Systems: stateless versus resident database lifecycle
bash scripts/profile_m5_nsys.sh \
    build-cuda/vector_search /tmp/vector-search-m5-nsys
```

Under WSL2, Nsight Compute performance counters may require enabling GPU
performance-counter access on the Windows host. The scripts report permission
errors and do not change host security settings.
