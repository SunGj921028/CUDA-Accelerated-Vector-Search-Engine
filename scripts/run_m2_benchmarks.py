#!/usr/bin/env python3
"""Run the M2 crossover, dimension, and stage-timing measurements.

The script intentionally uses only the Python standard library. The executable
prints key=value records, so this runner can preserve the benchmark metadata
alongside the measurements without adding a CSV dependency.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CROSSOVER_SIZES = (10_000, 15_000, 20_000, 40_000, 60_000, 80_000, 100_000)
DIMENSIONS = (128, 256, 768)
STAGE_WORKLOADS = (
    (10_000, 128),
    (100_000, 128),
    (100_000, 256),
    (100_000, 768),
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def default_binary() -> Path:
    for candidate in (
        REPOSITORY_ROOT / "build-cuda" / "vector_search",
        REPOSITORY_ROOT / "build" / "vector_search",
    ):
        if candidate.is_file():
            return candidate
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
    timing_breakdown: bool = False,
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
    if timing_breakdown:
        command.append("--timing-breakdown")

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


def required_int(values: Dict[str, str], key: str) -> int:
    if key not in values:
        raise RuntimeError("benchmark output did not contain {!r}".format(key))
    try:
        return int(values[key])
    except ValueError as error:
        raise RuntimeError(
            "benchmark output field {!r} was not an integer: {!r}".format(
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


def run_pair(
    cpu_binary: Path,
    cuda_binary: Path,
    num_vectors: int,
    dimension: int,
    num_queries: int,
    topk: int,
    seed: int,
    warmup: int,
    iterations: int,
) -> Dict[str, object]:
    cpu_values = run_search(
        cpu_binary,
        "cpu",
        num_vectors,
        dimension,
        num_queries,
        topk,
        seed,
        warmup,
        iterations,
    )
    cuda_values = run_search(
        cuda_binary,
        "cuda-naive",
        num_vectors,
        dimension,
        num_queries,
        topk,
        seed,
        warmup,
        iterations,
    )

    cpu_e2e = required_float(cpu_values, "latency_ms_average")
    cuda_e2e = required_float(cuda_values, "latency_ms_average")
    cuda_kernel = required_float(cuda_values, "kernel_latency_ms_average")
    cpu_checksum = required_float(cpu_values, "result_checksum")
    cuda_checksum = required_float(cuda_values, "result_checksum")

    row: Dict[str, object] = {
        "N": num_vectors,
        "D": dimension,
        "Q": num_queries,
        "K": topk,
        "seed": seed,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "cpu_e2e_ms": cpu_e2e,
        "cuda_kernel_ms": cuda_kernel,
        "cuda_e2e_ms": cuda_e2e,
        "speedup_cpu_over_cuda": cpu_e2e / cuda_e2e if cuda_e2e > 0.0 else 0.0,
        "cpu_checksum": cpu_checksum,
        "cuda_checksum": cuda_checksum,
        "checksum_abs_difference": abs(cpu_checksum - cuda_checksum),
        "cpu_min_ms": required_float(cpu_values, "latency_ms_min"),
        "cpu_max_ms": required_float(cpu_values, "latency_ms_max"),
        "cuda_kernel_min_ms": required_float(
            cuda_values, "kernel_latency_ms_min"
        ),
        "cuda_kernel_max_ms": required_float(
            cuda_values, "kernel_latency_ms_max"
        ),
        "cuda_e2e_min_ms": required_float(cuda_values, "latency_ms_min"),
        "cuda_e2e_max_ms": required_float(cuda_values, "latency_ms_max"),
    }
    row.update(metadata(cuda_values))
    return row


def run_stage_measurement(
    cuda_binary: Path,
    num_vectors: int,
    dimension: int,
    num_queries: int,
    topk: int,
    seed: int,
    warmup: int,
    iterations: int,
) -> Dict[str, object]:
    values = run_search(
        cuda_binary,
        "cuda-naive",
        num_vectors,
        dimension,
        num_queries,
        topk,
        seed,
        warmup,
        iterations,
        timing_breakdown=True,
    )
    if required_int(values, "stage_timed_iterations") != iterations:
        raise RuntimeError("not every measured iteration returned stage timing")

    row: Dict[str, object] = {
        "N": num_vectors,
        "D": dimension,
        "Q": num_queries,
        "K": topk,
        "seed": seed,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "allocation_ms": required_float(values, "allocation_ms_average"),
        "h2d_database_ms": required_float(
            values, "h2d_database_ms_average"
        ),
        "h2d_queries_ms": required_float(values, "h2d_queries_ms_average"),
        "kernel_ms": required_float(values, "kernel_ms_average"),
        "d2h_scores_ms": required_float(values, "d2h_scores_ms_average"),
        "cpu_topk_ms": required_float(values, "cpu_topk_ms_average"),
        "cleanup_ms": required_float(values, "cleanup_ms_average"),
        "total_e2e_ms": required_float(values, "total_e2e_ms_average"),
        "benchmark_e2e_ms": required_float(values, "latency_ms_average"),
    }
    row.update(metadata(values))
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


def print_summary(label: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    print(label)
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row[field]) for field in fields))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        help="Use one executable for CPU and CUDA runs (default: build-cuda/vector_search, then build/vector_search)",
    )
    parser.add_argument("--cpu-binary", type=Path)
    parser.add_argument("--cuda-binary", type=Path)
    parser.add_argument(
        "--mode",
        choices=("all", "crossover", "dimension", "stages"),
        default="all",
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
    shared_binary = resolve_path(arguments.binary) if arguments.binary else None
    cpu_binary = resolve_path(arguments.cpu_binary) if arguments.cpu_binary else shared_binary
    cuda_binary = resolve_path(arguments.cuda_binary) if arguments.cuda_binary else shared_binary
    if cpu_binary is None:
        cpu_binary = default_binary()
    if cuda_binary is None:
        cuda_binary = default_binary()

    for label, binary in (("CPU", cpu_binary), ("CUDA", cuda_binary)):
        if not binary.is_file():
            raise RuntimeError(
                "{} benchmark executable does not exist: {}".format(label, binary)
            )

    output_dir = resolve_path(arguments.output_dir)
    common = {
        "num_queries": arguments.queries,
        "topk": arguments.topk,
        "seed": arguments.seed,
        "warmup": arguments.warmup,
        "iterations": arguments.iterations,
    }

    if arguments.mode in ("all", "crossover"):
        crossover_rows = [
            run_pair(cpu_binary, cuda_binary, num_vectors, 128, **common)
            for num_vectors in CROSSOVER_SIZES
        ]
        crossover_path = output_dir / "m2_crossover.csv"
        write_csv(crossover_path, crossover_rows)
        print_summary(
            "crossover -> {}".format(crossover_path),
            crossover_rows,
            ("N", "cpu_e2e_ms", "cuda_kernel_ms", "cuda_e2e_ms", "speedup_cpu_over_cuda"),
        )

    if arguments.mode in ("all", "dimension"):
        dimension_rows = [
            run_pair(cpu_binary, cuda_binary, 100_000, dimension, **common)
            for dimension in DIMENSIONS
        ]
        dimension_path = output_dir / "m2_dimension.csv"
        write_csv(dimension_path, dimension_rows)
        print_summary(
            "dimension -> {}".format(dimension_path),
            dimension_rows,
            ("D", "cpu_e2e_ms", "cuda_kernel_ms", "cuda_e2e_ms", "speedup_cpu_over_cuda"),
        )

    if arguments.mode in ("all", "stages"):
        stage_rows = [
            run_stage_measurement(cuda_binary, num_vectors, dimension, **common)
            for num_vectors, dimension in STAGE_WORKLOADS
        ]
        stage_path = output_dir / "m2_stage_timing.csv"
        write_csv(stage_path, stage_rows)
        print_summary(
            "stages -> {}".format(stage_path),
            stage_rows,
            (
                "N",
                "D",
                "allocation_ms",
                "h2d_database_ms",
                "h2d_queries_ms",
                "kernel_ms",
                "d2h_scores_ms",
                "cpu_topk_ms",
                "cleanup_ms",
                "total_e2e_ms",
                "benchmark_e2e_ms",
            ),
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
