#pragma once

#include "vector_search.hpp"

#include <cstddef>
#include <vector>

namespace vector_search {

struct BenchmarkConfig {
    std::size_t warmup_iterations = 1;
    std::size_t measured_iterations = 5;
};

struct BenchmarkResult {
    std::size_t warmup_iterations = 0;
    std::size_t measured_iterations = 0;
    double min_milliseconds = 0.0;
    double average_milliseconds = 0.0;
    double max_milliseconds = 0.0;
    double queries_per_second = 0.0;
    double result_checksum = 0.0;
    bool has_kernel_timing = false;
    double kernel_min_milliseconds = 0.0;
    double kernel_average_milliseconds = 0.0;
    double kernel_max_milliseconds = 0.0;
    bool has_stage_timing = false;
    std::size_t stage_timed_iterations = 0;
    double allocation_average_milliseconds = 0.0;
    double h2d_database_average_milliseconds = 0.0;
    double h2d_queries_average_milliseconds = 0.0;
    double d2h_scores_average_milliseconds = 0.0;
    double cpu_topk_average_milliseconds = 0.0;
    double cleanup_average_milliseconds = 0.0;
    double total_e2e_average_milliseconds = 0.0;
    bool has_database_preparation_timing = false;
    double database_allocation_average_milliseconds = 0.0;
    double database_h2d_average_milliseconds = 0.0;
    double prepare_total_average_milliseconds = 0.0;
};

BenchmarkResult benchmark_search(
    const SearchBackend& backend,
    const SearchRequest& request,
    const BenchmarkConfig& config);

struct RepeatedBenchmarkConfig {
    std::size_t warmup_batches = 2;
    std::size_t measured_batches = 20;
};

struct RepeatedBenchmarkResult {
    std::size_t warmup_batches = 0;
    std::size_t measured_batches = 0;
    std::vector<double> measured_latencies_milliseconds;
    std::vector<BackendTiming> measured_timings;
    bool has_database_preparation_timing = false;
    BackendTiming database_preparation_timing{};
    double result_checksum = 0.0;
};

RepeatedBenchmarkResult benchmark_repeated_search(
    const SearchBackend& backend,
    const std::vector<SearchRequest>& requests,
    const RepeatedBenchmarkConfig& config);

}  // namespace vector_search
