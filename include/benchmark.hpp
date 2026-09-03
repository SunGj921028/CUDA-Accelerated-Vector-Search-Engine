#pragma once

#include "vector_search.hpp"

#include <cstddef>

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
};

BenchmarkResult benchmark_search(
    const SearchBackend& backend,
    const SearchRequest& request,
    const BenchmarkConfig& config);

}  // namespace vector_search
