#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys is not installed or is not on PATH; install NVIDIA Nsight Systems CLI." >&2
    exit 2
fi

if [[ $# -gt 2 ]]; then
    echo "usage: $0 [vector_search_binary] [output_directory]" >&2
    exit 2
fi

binary="${1:-build-cuda/vector_search}"
output_directory="${2:-/tmp/vector-search-nsys}"

if [[ ! -x "$binary" ]]; then
    echo "CUDA benchmark executable is not executable: $binary" >&2
    exit 2
fi

mkdir -p "$output_directory"

profile_workload() {
    local label="$1"
    local vectors="$2"
    local dimension="$3"
    local report_base="$output_directory/naive_${label}"

    nsys profile \
        --trace=cuda \
        --sample=none \
        --cpuctxsw=none \
        --cuda-event-trace=false \
        --output="$report_base" \
        --force-overwrite=true \
        "$binary" \
        --backend cuda-naive \
        --vectors "$vectors" \
        --dim "$dimension" \
        --queries 4 \
        --topk 10 \
        --seed 42 \
        --benchmark \
        --warmup 2 \
        --iterations 1

    nsys stats \
        --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
        --format column \
        --force-export=true \
        "${report_base}.nsys-rep" \
        > "${report_base}_summary.txt"

    echo "Nsight Systems report: ${report_base}.nsys-rep"
    echo "Nsight Systems summary: ${report_base}_summary.txt"
}

profile_workload n10000_d128 10000 128
profile_workload n100000_d768 100000 768
