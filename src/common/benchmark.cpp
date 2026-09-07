#include "benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vector_search {
namespace {

double checksum(const SearchResult& result) {
    double value = 0.0;
    for (const SearchHit& hit : result.hits) {
        value += static_cast<double>(hit.score) *
                 static_cast<double>(hit.index + 1);
    }
    return value;
}

}  // namespace

BenchmarkResult benchmark_search(
    const SearchBackend& backend,
    const SearchRequest& request,
    const BenchmarkConfig& config) {
    if (config.measured_iterations == 0) {
        throw std::invalid_argument("measured_iterations must be greater than zero");
    }

    double warmup_checksum = 0.0;
    for (std::size_t iteration = 0;
         iteration < config.warmup_iterations;
         ++iteration) {
        const SearchResult result = backend.search(request);
        warmup_checksum += checksum(result);
    }

    double total_milliseconds = 0.0;
    double min_milliseconds = std::numeric_limits<double>::max();
    double max_milliseconds = 0.0;
    double result_checksum = warmup_checksum;
    bool has_kernel_timing = false;
    double kernel_total_milliseconds = 0.0;
    double kernel_min_milliseconds = std::numeric_limits<double>::max();
    double kernel_max_milliseconds = 0.0;

    for (std::size_t iteration = 0;
         iteration < config.measured_iterations;
         ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        const SearchResult result = backend.search(request);
        const auto stop = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double, std::milli>(stop - start).count();

        total_milliseconds += elapsed;
        min_milliseconds = std::min(min_milliseconds, elapsed);
        max_milliseconds = std::max(max_milliseconds, elapsed);
        result_checksum += checksum(result);

        const BackendTiming timing = backend.last_timing();
        if (timing.has_kernel_latency) {
            if (!std::isfinite(timing.kernel_latency_milliseconds) ||
                timing.kernel_latency_milliseconds < 0.0) {
                throw std::runtime_error(
                    "backend reported an invalid kernel latency");
            }
            has_kernel_timing = true;
            kernel_total_milliseconds += timing.kernel_latency_milliseconds;
            kernel_min_milliseconds = std::min(
                kernel_min_milliseconds, timing.kernel_latency_milliseconds);
            kernel_max_milliseconds = std::max(
                kernel_max_milliseconds, timing.kernel_latency_milliseconds);
        }
    }

    const double average_milliseconds =
        total_milliseconds / static_cast<double>(config.measured_iterations);

    BenchmarkResult benchmark;
    benchmark.warmup_iterations = config.warmup_iterations;
    benchmark.measured_iterations = config.measured_iterations;
    benchmark.min_milliseconds = min_milliseconds;
    benchmark.average_milliseconds = average_milliseconds;
    benchmark.max_milliseconds = max_milliseconds;
    benchmark.queries_per_second = average_milliseconds > 0.0
                                       ? static_cast<double>(request.num_queries) /
                                             (average_milliseconds / 1000.0)
                                       : 0.0;
    benchmark.result_checksum = result_checksum;
    benchmark.has_kernel_timing = has_kernel_timing;
    if (has_kernel_timing) {
        benchmark.kernel_min_milliseconds = kernel_min_milliseconds;
        benchmark.kernel_average_milliseconds =
            kernel_total_milliseconds /
            static_cast<double>(config.measured_iterations);
        benchmark.kernel_max_milliseconds = kernel_max_milliseconds;
    }
    return benchmark;
}

}  // namespace vector_search
