# AGENTS.md

## Project Purpose

This repository implements and analyzes a CUDA-accelerated exact vector search engine.

The project is primarily an educational and performance-engineering project.

Correctness, reproducibility, understandable implementations, and evidence-driven optimization are more important than minimizing code size.

## Core Technologies

* C++17 or newer
* CUDA C++
* CMake
* Python only for benchmark orchestration / result processing
* NVIDIA Nsight Systems
* NVIDIA Nsight Compute

## Development Principles

### 1. Do not optimize without evidence

Do not introduce a performance optimization unless:

1. a baseline exists,
2. the current implementation is benchmarked,
3. a performance hypothesis can be stated,
4. the result can be measured.

### 2. Preserve implementation stages

Do not replace previous CUDA implementations when creating optimized versions.

Keep:

* naive implementation
* intermediate implementations
* optimized implementation

as separate backends whenever practical.

This repository is intended to demonstrate the optimization progression.

### 3. CPU implementation is the correctness reference

Every CUDA backend must be validated against the CPU reference implementation.

Floating-point comparisons must use an appropriate tolerance.

### 4. Do not hide core CUDA logic behind external libraries

For the core learning milestones:

* similarity kernels must be implemented directly in CUDA C++.
* core parallel-reduction logic should be implemented explicitly.

Optimized libraries such as cuBLAS, CUB, or Thrust may be added later as reference implementations or optional backends.

### 5. Benchmark honestly

Distinguish between:

* kernel-only latency
* end-to-end latency

Never report a performance improvement without specifying what was measured.

Use warm-up executions and multiple measured iterations.

### 6. Reproducibility

Benchmark output must record:

* CPU
* GPU
* CUDA version
* compiler version
* dataset size
* vector dimension
* query count
* Top-K
* backend
* timing results

### 7. CUDA error handling

Every CUDA API operation and kernel launch must be checked for errors.

Do not ignore CUDA errors.

### 8. Keep code readable

Prefer:

* clear kernels
* explicit naming
* comments explaining CUDA-specific decisions
* small modules

over clever but opaque optimization code.

## Required Validation Before Completing a Task

For every implementation change:

1. configure/build the project,
2. run relevant tests,
3. run correctness comparison against the CPU backend,
4. report commands executed,
5. report test/build results,
6. clearly state anything that could not be tested.

For performance-related changes:

also report the benchmark configuration and before/after measurements.

## Documentation

When implementing a new optimization, update:

```
docs/optimization_log.md
```

with:

* baseline
* hypothesis
* implementation change
* benchmark result
* interpretation
* decision

Do not claim an optimization improved performance unless measurements demonstrate it.

## Project Scope

Do not add the following unless explicitly requested:

* RAG frameworks
* LLM APIs
* vector databases
* frontend applications
* web servers
* distributed systems
* multi-GPU support
* approximate nearest-neighbor algorithms

Keep the project focused on CUDA and GPU performance engineering.
