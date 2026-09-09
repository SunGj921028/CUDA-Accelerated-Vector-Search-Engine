#!/usr/bin/env bash
set -euo pipefail

ncu_binary="$(command -v ncu || true)"
if [[ -z "$ncu_binary" && -x "/opt/nvidia/nsight-compute/2025.2.1/ncu" ]]; then
    ncu_binary="/opt/nvidia/nsight-compute/2025.2.1/ncu"
fi
if [[ -z "$ncu_binary" ]]; then
    echo "ncu is not installed; install the NVIDIA Nsight Compute CLI." >&2
    exit 2
fi

if [[ $# -gt 5 ]]; then
    echo "usage: $0 [vector_search_binary] [output_directory] [vectors] [dimension] [backend]" >&2
    exit 2
fi

binary="${1:-build-cuda/vector_search}"
output_directory="${2:-profiling/ncu}"
vectors="${3:-100000}"
dimension="${4:-768}"
backend="${5:-cuda-naive}"

case "$backend" in
    cuda-naive)
        kernel_name="naive_similarity_kernel"
        report_prefix="m2"
        ;;
    cuda-block)
        kernel_name="block_similarity_kernel"
        report_prefix="m3_cuda-block"
        ;;
    cuda-warp)
        kernel_name="warp_similarity_kernel"
        report_prefix="m4_cuda-warp"
        ;;
    *)
        echo "unsupported CUDA backend: $backend (expected cuda-naive, cuda-block, or cuda-warp)" >&2
        exit 2
        ;;
esac

report_base="$output_directory/${report_prefix}_n${vectors}_d${dimension}"

if [[ ! -x "$binary" ]]; then
    echo "CUDA benchmark executable is not executable: $binary" >&2
    exit 2
fi

mkdir -p "$output_directory"

echo "Profiling one ${kernel_name} launch for backend ${backend} at N=${vectors}, D=${dimension}."
echo "If performance-counter permission errors occur under WSL2, enable GPU"
echo "performance-counter access on the Windows host; this script does not"
echo "change security settings or use privileged workarounds."
echo "Nsight Compute: $($ncu_binary --version | sed -n 's/^Version /Version /p' | head -1)"

"$ncu_binary" \
    --target-processes all \
    --kernel-name "regex:${kernel_name}" \
    --launch-skip 2 \
    --launch-count 1 \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables \
    --section Occupancy \
    --section LaunchStats \
    --section SchedulerStats \
    --section InstructionStats \
    --section WarpStateStats \
    --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld,smsp__sass_average_data_bytes_per_sector_mem_global_op_st,smsp__warp_issue_stalled_lg_throttle_per_warp_active,smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active \
    --csv \
    --export "$report_base" \
    --force-overwrite \
    "$binary" \
    --backend "$backend" \
    --vectors "$vectors" \
    --dim "$dimension" \
    --queries 4 \
    --topk 10 \
    --seed 42 \
    --benchmark \
    --warmup 2 \
    --iterations 1

"$ncu_binary" \
    --import "${report_base}.ncu-rep" \
    --page details \
    --csv \
    --log-file "${report_base}_details.csv"

echo "Nsight Compute report: ${report_base}.ncu-rep"
echo "Nsight Compute CSV summary: ${report_base}_details.csv"
