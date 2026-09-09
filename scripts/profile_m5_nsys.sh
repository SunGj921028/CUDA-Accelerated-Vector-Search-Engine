#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "usage: $0 [vector_search_binary] [output_directory]" >&2
    exit 2
fi

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${1:-$script_directory/../build-m5-cuda-wsl/vector_search}"
output_directory="${2:-$script_directory/../profiling/nsys/m5}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys is not installed or is not on PATH; install NVIDIA Nsight Systems CLI." >&2
    exit 2
fi

if [[ ! -x "$binary" ]]; then
    echo "CUDA benchmark executable is not executable: $binary" >&2
    exit 2
fi

mkdir -p "$output_directory"

for backend in cuda-warp cuda-warp-resident; do
    report_base="$output_directory/m5_${backend}_n10000_d128_q2"
    nsys profile \
        --trace=cuda \
        --sample=none \
        --cpuctxsw=none \
        --cuda-event-trace=true \
        --output="$report_base" \
        --force-overwrite=true \
        "$binary" \
        --backend "$backend" \
        --vectors 10000 \
        --dim 128 \
        --queries 2 \
        --topk 3 \
        --seed 42 \
        --timing-breakdown \
        --warmup 0 \
        --repeat-batches 3

    nsys stats \
        --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
        --format column \
        --force-export=true \
        "${report_base}.nsys-rep" \
        > "${report_base}_summary.txt"

    echo "Nsight Systems report: ${report_base}.nsys-rep"
    echo "Nsight Systems summary: ${report_base}_summary.txt"
done




