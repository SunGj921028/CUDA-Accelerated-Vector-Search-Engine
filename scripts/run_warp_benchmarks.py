#!/usr/bin/env python3
"""Run the controlled CPU, cuda-naive, cuda-block, and cuda-warp benchmarks."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_WORKLOAD = ("primary_100k_d768", 100_000, 768)
DIMENSION_WORKLOADS = (
    ("dimension_d32", 100_000, 32),
    ("dimension_d64", 100_000, 64),
    ("dimension_d128", 100_000, 128),
    ("dimension_d256", 100_000, 256),
    ("dimension_d512", 100_000, 512),
    ("dimension_d768", 100_000, 768),
    ("dimension_d1024", 100_000, 1024),
)
SMALL_WORKLOAD = ("small_10k_d128", 10_000, 128)
BACKENDS = ("cpu", "cuda-naive", "cuda-block", "cuda-warp")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def default_binary() -> Path:
    return REPOSITORY_ROOT / "build-cuda" / "vector_search"


def parse_key_value_output(output: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def run_search(
    binary: Path,
    backend: str,
    num_vectors: int,
    dimension: int,
    num_queries: int,
    topk: int,
    seed: int,
    warmup: int,
    iterations: int,
) -> Dict[str, str]:
    command = [
        str(binary),
        "--backend",
        backend,
        "--vectors",
        str(num_vectors),
        "--dim",
        str(dimension),
        "--queries",
        str(num_queries),
        "--topk",
        str(topk),
        "--seed",
        str(seed),
        "--benchmark",
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "benchmark command failed (exit code {}):\n{}\n{}".format(
                completed.returncode,
                " ".join(command),
                completed.stdout + completed.stderr,
            )
        )
    return parse_key_value_output(completed.stdout)


def required_float(values: Dict[str, str], key: str) -> float:
    if key not in values:
        raise RuntimeError("benchmark output did not contain {!r}".format(key))
    try:
        return float(values[key])
    except ValueError as error:
        raise RuntimeError(
            "benchmark output field {!r} was not numeric: {!r}".format(
                key, values[key]
            )
        ) from error


def metadata(values: Dict[str, str]) -> Dict[str, str]:
    return {
        "cpu_logical_processors": values.get("cpu_logical_processors", "unknown"),
        "gpu": values.get("gpu", "unavailable"),
        "cuda_toolkit": values.get("cuda_toolkit", "unavailable"),
        "compiler": values.get("compiler", "unknown"),
    }


def run_workload(
    binary: Path,
    label: str,
    num_vectors: int,
    dimension: int,
    num_queries: int,
    topk: int,
    seed: int,
    warmup: int,
    iterations: int,
) -> Dict[str, object]:
    values = {
        backend: run_search(
            binary,
            backend,
            num_vectors,
            dimension,
            num_queries,
            topk,
            seed,
            warmup,
            iterations,
        )
        for backend in BACKENDS
    }

    kernels = {
        backend: required_float(values[backend], "kernel_latency_ms_average")
        for backend in ("cuda-naive", "cuda-block", "cuda-warp")
    }
    e2e = {
        backend: required_float(values[backend], "latency_ms_average")
        for backend in ("cuda-naive", "cuda-block", "cuda-warp")
    }

    row: Dict[str, object] = {
        "workload": label,
        "N": num_vectors,
        "D": dimension,
        "Q": num_queries,
        "K": topk,
        "seed": seed,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "cpu_e2e_ms": required_float(values["cpu"], "latency_ms_average"),
        "naive_kernel_ms": kernels["cuda-naive"],
        "block_kernel_ms": kernels["cuda-block"],
        "warp_kernel_ms": kernels["cuda-warp"],
        "warp_vs_naive_kernel_speedup": (
            kernels["cuda-naive"] / kernels["cuda-warp"]
            if kernels["cuda-warp"] > 0.0
            else 0.0
        ),
        "warp_vs_block_kernel_speedup": (
            kernels["cuda-block"] / kernels["cuda-warp"]
            if kernels["cuda-warp"] > 0.0
            else 0.0
        ),
        "naive_e2e_ms": e2e["cuda-naive"],
        "block_e2e_ms": e2e["cuda-block"],
        "warp_e2e_ms": e2e["cuda-warp"],
        "warp_vs_naive_e2e_speedup": (
            e2e["cuda-naive"] / e2e["cuda-warp"]
            if e2e["cuda-warp"] > 0.0
            else 0.0
        ),
        "warp_vs_block_e2e_speedup": (
            e2e["cuda-block"] / e2e["cuda-warp"]
            if e2e["cuda-warp"] > 0.0
            else 0.0
        ),
    }
    for backend in ("cpu", "cuda-naive", "cuda-block", "cuda-warp"):
        row[f"{backend.replace('-', '_')}_checksum"] = required_float(
            values[backend], "result_checksum"
        )
    row["cpu_naive_checksum_abs_difference"] = abs(
        float(row["cpu_checksum"]) - float(row["cuda_naive_checksum"])
    )
    row["cpu_block_checksum_abs_difference"] = abs(
        float(row["cpu_checksum"]) - float(row["cuda_block_checksum"])
    )
    row["cpu_warp_checksum_abs_difference"] = abs(
        float(row["cpu_checksum"]) - float(row["cuda_warp_checksum"])
    )
    row.update(metadata(values["cuda-warp"]))
    return row


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    row_list = list(rows)
    if not row_list:
        return
    fieldnames = list(row_list[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    fields = (
        "workload",
        "N",
        "D",
        "cpu_e2e_ms",
        "naive_kernel_ms",
        "block_kernel_ms",
        "warp_kernel_ms",
        "warp_vs_naive_kernel_speedup",
        "warp_vs_block_kernel_speedup",
        "naive_e2e_ms",
        "block_e2e_ms",
        "warp_e2e_ms",
        "warp_vs_naive_e2e_speedup",
        "warp_vs_block_e2e_speedup",
    )
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row[field]) for field in fields))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=default_binary(),
        help="CUDA benchmark executable",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
    )
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    binary = resolve_path(arguments.binary)
    if not binary.is_file():
        raise RuntimeError(
            "CUDA benchmark executable does not exist: {}".format(binary)
        )

    common = {
        "num_queries": arguments.queries,
        "topk": arguments.topk,
        "seed": arguments.seed,
        "warmup": arguments.warmup,
        "iterations": arguments.iterations,
    }
    primary = [run_workload(binary, *PRIMARY_WORKLOAD, **common)]
    dimensions = [
        run_workload(binary, *workload, **common)
        for workload in DIMENSION_WORKLOADS
    ]
    small = [run_workload(binary, *SMALL_WORKLOAD, **common)]
    all_rows = primary + dimensions + small

    output_dir = resolve_path(arguments.output_dir)
    write_csv(output_dir / "warp_primary.csv", primary)
    write_csv(output_dir / "warp_dimension.csv", dimensions)
    write_csv(output_dir / "warp_small_workload.csv", small)
    write_csv(output_dir / "warp_comparison.csv", all_rows)
    print_summary(all_rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
