#!/usr/bin/env python3
"""Run the controlled stateless-versus-GPU-resident benchmark."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_WORKLOAD = ("primary_100k_d768", 100_000, 768)
DIMENSION_WORKLOADS = (
    ("dimension_d128", 100_000, 128),
    ("dimension_d256", 100_000, 256),
    ("dimension_d768", 100_000, 768),
)
OPTIONAL_WORKLOAD = ("optional_250k_d768", 250_000, 768)
BACKENDS = ("cuda-warp", "cuda-warp-resident")
AMORTIZATION_BATCHES = (1, 2, 5, 10, 20, 100)


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


def run_command(command: Sequence[str]) -> Dict[str, str]:
    completed = subprocess.run(
        list(command),
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


def run_cold(
    binary: Path,
    backend: str,
    num_vectors: int,
    dimension: int,
    num_queries: int,
    topk: int,
    seed: int,
) -> Dict[str, str]:
    return run_command(
        (
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
            "--timing-breakdown",
            "--warmup",
            "0",
            "--iterations",
            "1",
        )
    )


def run_repeated(
    binary: Path,
    backend: str,
    num_vectors: int,
    dimension: int,
    num_queries: int,
    topk: int,
    seed: int,
    warmup: int,
    measured_batches: int,
) -> Dict[str, str]:
    return run_command(
        (
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
            "--timing-breakdown",
            "--warmup",
            str(warmup),
            "--repeat-batches",
            str(measured_batches),
        )
    )


def required_float(values: Dict[str, str], key: str) -> float:
    if key not in values:
        raise RuntimeError("benchmark output did not contain {!r}".format(key))
    try:
        value = float(values[key])
    except ValueError as error:
        raise RuntimeError(
            "benchmark output field {!r} was not numeric: {!r}".format(
                key, values[key]
            )
        ) from error
    if value < 0.0:
        raise RuntimeError("benchmark output field {!r} was negative".format(key))
    return value


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


def optional_float(values: Dict[str, str], key: str, default: float = 0.0) -> float:
    return required_float(values, key) if key in values else default


def batch_values(
    values: Dict[str, str], measured_batches: int, suffix: str
) -> List[float]:
    result = [
        required_float(values, "batch_{}_{}".format(index, suffix))
        for index in range(1, measured_batches + 1)
    ]
    return result


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def median(values: Sequence[float]) -> float:
    return statistics.median(values)


def safe_speedup(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def metadata(values: Dict[str, str]) -> Dict[str, str]:
    return {
        "cpu_logical_processors": values.get("cpu_logical_processors", "unknown"),
        "gpu": values.get("gpu", "unavailable"),
        "cuda_toolkit": values.get("cuda_toolkit", "unavailable"),
        "compiler": values.get("compiler", "unknown"),
    }


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


def workload_metadata(
    label: str,
    num_vectors: int,
    dimension: int,
    num_queries: int,
    topk: int,
    seed: int,
    warmup: int,
    measured_batches: int,
) -> Dict[str, object]:
    return {
        "workload": label,
        "N": num_vectors,
        "D": dimension,
        "Q": num_queries,
        "K": topk,
        "seed": seed,
        "warmup_batches": warmup,
        "measured_batches": measured_batches,
    }


def run_cold_workload(
    binary: Path,
    workload: Tuple[str, int, int],
    num_queries: int,
    topk: int,
    seed: int,
) -> Dict[str, object]:
    label, num_vectors, dimension = workload
    stateless = run_cold(
        binary, "cuda-warp", num_vectors, dimension, num_queries, topk, seed
    )
    resident = run_cold(
        binary,
        "cuda-warp-resident",
        num_vectors,
        dimension,
        num_queries,
        topk,
        seed,
    )
    stateless_e2e = required_float(stateless, "latency_ms_average")
    resident_query = required_float(resident, "latency_ms_average")
    resident_prepare = required_float(resident, "prepare_total_ms_average")
    resident_db_allocation = required_float(
        resident, "db_allocation_ms_average")
    resident_db_h2d = required_float(resident, "db_h2d_ms_average")
    resident_cold = optional_float(
        resident,
        "cold_start_prepare_plus_first_query_ms",
        resident_prepare + resident_query,
    )
    checksum_difference = abs(
        required_float(stateless, "result_checksum")
        - required_float(resident, "result_checksum")
    )
    row: Dict[str, object] = workload_metadata(
        label, num_vectors, dimension, num_queries, topk, seed, 0, 1
    )
    row.update(
        {
            "database_bytes": required_int(resident, "database_bytes"),
            "database_mib": required_float(resident, "database_mib"),
            "resident_db_allocation_ms": resident_db_allocation,
            "resident_db_h2d_ms": resident_db_h2d,
            "device_free_bytes_before_prepare": required_int(
                resident, "device_free_bytes_before_prepare"
            ),
            "device_free_bytes_after_prepare": required_int(
                resident, "device_free_bytes_after_prepare"
            ),
            "cuda_warp_e2e_ms": stateless_e2e,
            "resident_prepare_total_ms": resident_prepare,
            "resident_first_query_ms": resident_query,
            "resident_cold_total_ms": resident_cold,
            "resident_vs_cuda_warp_cold_speedup": safe_speedup(
                stateless_e2e, resident_cold
            ),
            "cuda_warp_checksum": required_float(stateless, "result_checksum"),
            "resident_checksum": required_float(resident, "result_checksum"),
            "checksum_abs_difference": checksum_difference,
        }
    )
    row.update(metadata(resident))
    return row


def run_warm_workload(
    binary: Path,
    workload: Tuple[str, int, int],
    num_queries: int,
    topk: int,
    seed: int,
    warmup: int,
    measured_batches: int,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, str]]]:
    label, num_vectors, dimension = workload
    outputs = {
        backend: run_repeated(
            binary,
            backend,
            num_vectors,
            dimension,
            num_queries,
            topk,
            seed,
            warmup,
            measured_batches,
        )
        for backend in BACKENDS
    }
    stateless = outputs["cuda-warp"]
    resident = outputs["cuda-warp-resident"]
    stateless_latencies = batch_values(stateless, measured_batches, "latency_ms")
    resident_latencies = batch_values(resident, measured_batches, "latency_ms")
    stateless_kernels = batch_values(stateless, measured_batches, "kernel_ms")
    resident_kernels = batch_values(resident, measured_batches, "kernel_ms")
    resident_stage = {
        name: batch_values(resident, measured_batches, "query_{}_ms".format(name))
        for name in (
            "allocation",
            "h2d",
            "d2h_scores",
            "cpu_topk",
            "cleanup",
            "e2e",
        )
    }
    stateless_db_h2d = batch_values(
        stateless, measured_batches, "h2d_database_ms"
    )
    stateless_stage = {
        name: batch_values(stateless, measured_batches, "{}_ms".format(name))
        for name in (
            "allocation",
            "h2d_queries",
            "d2h_scores",
            "cpu_topk",
            "cleanup",
            "e2e",
        )
    }
    checksum_difference = abs(
        required_float(stateless, "result_checksum")
        - required_float(resident, "result_checksum")
    )
    row: Dict[str, object] = workload_metadata(
        label,
        num_vectors,
        dimension,
        num_queries,
        topk,
        seed,
        warmup,
        measured_batches,
    )
    row.update(
        {
            "database_bytes": required_int(resident, "database_bytes"),
            "database_mib": required_float(resident, "database_mib"),
            "resident_db_allocation_ms": required_float(
                resident, "db_allocation_ms"
            ),
            "resident_db_h2d_ms": required_float(resident, "db_h2d_ms"),
            "device_free_bytes_before_prepare": required_int(
                resident, "device_free_bytes_before_prepare"
            ),
            "device_free_bytes_after_prepare": required_int(
                resident, "device_free_bytes_after_prepare"
            ),
            "resident_prepare_total_ms": required_float(
                resident, "prepare_total_ms"
            ),
            "stateless_mean_ms": mean(stateless_latencies),
            "stateless_median_ms": median(stateless_latencies),
            "resident_mean_ms": mean(resident_latencies),
            "resident_median_ms": median(resident_latencies),
            "resident_warm_speedup_mean": safe_speedup(
                mean(stateless_latencies), mean(resident_latencies)
            ),
            "resident_warm_speedup_median": safe_speedup(
                median(stateless_latencies), median(resident_latencies)
            ),
            "stateless_kernel_mean_ms": mean(stateless_kernels),
            "resident_kernel_mean_ms": mean(resident_kernels),
            "stateless_db_h2d_mean_ms": mean(stateless_db_h2d),
            "stateless_allocation_mean_ms": mean(stateless_stage["allocation"]),
            "stateless_query_h2d_mean_ms": mean(stateless_stage["h2d_queries"]),
            "stateless_d2h_scores_mean_ms": mean(stateless_stage["d2h_scores"]),
            "stateless_cpu_topk_mean_ms": mean(stateless_stage["cpu_topk"]),
            "stateless_cleanup_mean_ms": mean(stateless_stage["cleanup"]),
            "stateless_e2e_stage_mean_ms": mean(stateless_stage["e2e"]),
            "resident_query_allocation_mean_ms": mean(
                resident_stage["allocation"]
            ),
            "resident_query_h2d_mean_ms": mean(resident_stage["h2d"]),
            "resident_query_d2h_scores_mean_ms": mean(
                resident_stage["d2h_scores"]
            ),
            "resident_query_cpu_topk_mean_ms": mean(resident_stage["cpu_topk"]),
            "resident_query_cleanup_mean_ms": mean(resident_stage["cleanup"]),
            "resident_query_e2e_stage_mean_ms": mean(resident_stage["e2e"]),
            "cuda_warp_checksum": required_float(
                stateless, "result_checksum"
            ),
            "resident_checksum": required_float(resident, "result_checksum"),
            "checksum_abs_difference": checksum_difference,
        }
    )
    row.update(metadata(resident))
    return row, outputs


def build_amortization_rows(
    workload: Tuple[str, int, int],
    num_queries: int,
    topk: int,
    seed: int,
    warmup: int,
    measured_batches: int,
    outputs: Dict[str, Dict[str, str]],
    prepare_total_ms: float,
) -> List[Dict[str, object]]:
    label, num_vectors, dimension = workload
    stateless_latencies = batch_values(
        outputs["cuda-warp"], measured_batches, "latency_ms"
    )
    resident_latencies = batch_values(
        outputs["cuda-warp-resident"], measured_batches, "latency_ms"
    )
    rows: List[Dict[str, object]] = []
    for batch_count in AMORTIZATION_BATCHES:
        if batch_count > measured_batches:
            continue
        stateless_total = sum(stateless_latencies[:batch_count])
        resident_queries_total = sum(resident_latencies[:batch_count])
        resident_total = prepare_total_ms + resident_queries_total
        row: Dict[str, object] = workload_metadata(
            label,
            num_vectors,
            dimension,
            num_queries,
            topk,
            seed,
            warmup,
            batch_count,
        )
        row.update(
            {
                "resident_prepare_total_ms": prepare_total_ms,
                "stateless_total_ms": stateless_total,
                "stateless_average_ms": stateless_total / batch_count,
                "resident_query_total_ms": resident_queries_total,
                "resident_total_amortized_ms": resident_total,
                "resident_average_amortized_ms": resident_total / batch_count,
                "resident_total_speedup": safe_speedup(
                    stateless_total, resident_total
                ),
                "resident_average_speedup": safe_speedup(
                    stateless_total / batch_count,
                    resident_total / batch_count,
                ),
            }
        )
        rows.append(row)
    return rows


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
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="measured repeated query batches; must cover 100 for amortization",
    )
    parser.add_argument(
        "--include-250k",
        action="store_true",
        help="also run the optional 250K x 768 workload",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.queries <= 0 or arguments.topk <= 0:
        raise RuntimeError("--queries and --topk must be greater than zero")
    if arguments.warmup < 0 or arguments.iterations <= 0:
        raise RuntimeError("--warmup must be non-negative and iterations > 0")
    if arguments.include_250k and arguments.iterations < max(AMORTIZATION_BATCHES):
        raise RuntimeError(
            "--iterations must be at least 100 for the requested amortization table"
        )

    binary = resolve_path(arguments.binary)
    if not binary.is_file():
        raise RuntimeError(
            "CUDA benchmark executable does not exist: {}".format(binary)
        )
    output_dir = resolve_path(arguments.output_dir)
    workloads = [PRIMARY_WORKLOAD, *DIMENSION_WORKLOADS]
    if arguments.include_250k:
        workloads.append(OPTIONAL_WORKLOAD)

    cold_rows = [
        run_cold_workload(
            binary, workload, arguments.queries, arguments.topk, arguments.seed
        )
        for workload in workloads
    ]

    warm_rows: List[Dict[str, object]] = []
    amortization_rows: List[Dict[str, object]] = []
    for workload in workloads:
        row, outputs = run_warm_workload(
            binary,
            workload,
            arguments.queries,
            arguments.topk,
            arguments.seed,
            arguments.warmup,
            arguments.iterations,
        )
        warm_rows.append(row)
        amortization_rows.extend(
            build_amortization_rows(
                workload,
                arguments.queries,
                arguments.topk,
                arguments.seed,
                arguments.warmup,
                arguments.iterations,
                outputs,
                float(row["resident_prepare_total_ms"]),
            )
        )

    write_csv(output_dir / "resident_cold_start.csv", cold_rows)
    write_csv(output_dir / "resident_amortization.csv", amortization_rows)
    write_csv(output_dir / "resident_comparison.csv", warm_rows)

    print("Resident cold-start results")
    print("workload,cuda_warp_e2e_ms,resident_db_allocation_ms,resident_db_h2d_ms,resident_prepare_ms,resident_first_query_ms,resident_cold_total_ms")
    for row in cold_rows:
        print(
            "{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(
                row["workload"],
                row["cuda_warp_e2e_ms"],
                row["resident_db_allocation_ms"],
                row["resident_db_h2d_ms"],
                row["resident_prepare_total_ms"],
                row["resident_first_query_ms"],
                row["resident_cold_total_ms"],
            )
        )

    print("Resident steady-state results")
    print("workload,stateless_mean_ms,resident_mean_ms,warm_speedup")
    for row in warm_rows:
        print(
            "{},{:.3f},{:.3f},{:.3f}".format(
                row["workload"],
                row["stateless_mean_ms"],
                row["resident_mean_ms"],
                row["resident_warm_speedup_mean"],
            )
        )
    print("Wrote {}".format(output_dir))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit("error: {}".format(error))


