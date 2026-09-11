#!/usr/bin/env python3
"""Render final kernel-latency and speedup figures from M4 CSV evidence."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_DIMENSIONS = (128, 256, 768)
BACKENDS = (
    ("naive_kernel_ms", "cuda-naive", "#4B5563", ""),
    ("block_kernel_ms", "cuda-block", "#2563EB", "//"),
    ("warp_kernel_ms", "cuda-warp", "#D97706", "xx"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/results/m4_dimension.csv",
        help="M4 dimension-sweep CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "docs/images",
        help="Directory for generated PNG files",
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    selected: Dict[int, Dict[str, str]] = {}
    for row in rows:
        dimension = int(row["D"])
        if dimension not in REPRESENTATIVE_DIMENSIONS:
            continue
        if dimension in selected:
            raise RuntimeError(f"duplicate D={dimension} row in {path}")
        selected[dimension] = row

    missing = set(REPRESENTATIVE_DIMENSIONS) - set(selected)
    if missing:
        raise RuntimeError(
            f"missing representative dimensions: {sorted(missing)}"
        )

    ordered = [selected[dimension] for dimension in REPRESENTATIVE_DIMENSIONS]
    for row in ordered:
        metadata = (
            int(row["N"]),
            int(row["Q"]),
            int(row["K"]),
            int(row["seed"]),
        )
        if metadata != (100_000, 4, 10, 42):
            raise RuntimeError("unexpected workload metadata in M4 dimension CSV")
        for field, _, _, _ in BACKENDS:
            value = float(row[field])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(f"invalid {field} value: {value}")
    return ordered


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "text.color": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )


def add_source_note(figure: plt.Figure) -> None:
    figure.text(
        0.5,
        0.015,
        "Source: benchmarks/results/m4_dimension.csv | "
        "N=100,000, Q=4, K=10, seed=42; "
        "2 warm-ups, mean of 5 measured iterations",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )


def render_latency(rows: List[Dict[str, str]], output_path: Path) -> None:
    dimensions = [int(row["D"]) for row in rows]
    centers = list(range(len(dimensions)))
    width = 0.24

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    for backend_index, (field, label, color, hatch) in enumerate(BACKENDS):
        positions = [
            center + (backend_index - 1) * width for center in centers
        ]
        values = [float(row[field]) for row in rows]
        bars = axis.bar(
            positions,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="#111827",
            linewidth=0.5,
            hatch=hatch,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in values],
            padding=3,
            fontsize=8.5,
        )

    axis.set_title("Kernel latency by CUDA backend")
    axis.set_ylabel("Kernel latency (ms, log scale)")
    axis.set_xlabel("Vector dimension (D)")
    axis.set_xticks(centers, [str(value) for value in dimensions])
    axis.set_yscale("log")
    axis.set_ylim(0.8, 100)
    axis.grid(axis="y", which="both", alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", ncols=3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    add_source_note(figure)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def render_speedup(rows: List[Dict[str, str]], output_path: Path) -> None:
    dimensions = [int(row["D"]) for row in rows]
    centers = list(range(len(dimensions)))
    width = 0.34
    series = (
        ("block_kernel_ms", "cuda-block", "#2563EB", "//"),
        ("warp_kernel_ms", "cuda-warp", "#D97706", "xx"),
    )

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    for series_index, (field, label, color, hatch) in enumerate(series):
        positions = [
            center + (series_index - 0.5) * width for center in centers
        ]
        values = [
            float(row["naive_kernel_ms"]) / float(row[field]) for row in rows
        ]
        bars = axis.bar(
            positions,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="#111827",
            linewidth=0.5,
            hatch=hatch,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.2f}x" for value in values],
            padding=3,
            fontsize=9,
        )

    axis.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
    axis.text(
        0.01,
        1.12,
        "1.0x = naive baseline",
        transform=axis.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    axis.set_title("Kernel speedup relative to cuda-naive")
    axis.set_ylabel("Speedup (naive kernel / backend kernel)")
    axis.set_xlabel("Vector dimension (D)")
    axis.set_xticks(centers, [str(value) for value in dimensions])
    axis.set_ylim(0, 10.5)
    axis.grid(axis="y", alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", ncols=2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    add_source_note(figure)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    input_path = arguments.input.resolve()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_path)
    apply_style()
    latency_path = output_dir / "kernel_latency_by_backend.png"
    speedup_path = output_dir / "kernel_speedup_vs_naive.png"
    render_latency(rows, latency_path)
    render_speedup(rows, speedup_path)
    print(latency_path)
    print(speedup_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
