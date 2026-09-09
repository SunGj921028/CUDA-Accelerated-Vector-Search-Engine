#include "benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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

void validate_timing(const BackendTiming& timing) {
    const auto validate = [](double value, const char* name) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error(
                std::string("backend reported an invalid ") + name);
        }
    };

    if (timing.has_kernel_latency) {
        validate(timing.kernel_latency_milliseconds, "kernel latency");
    }
    if (timing.has_stage_timing) {
        validate(timing.allocation_milliseconds, "allocation time");
        validate(timing.h2d_database_milliseconds, "H2D database time");
        validate(timing.h2d_queries_milliseconds, "H2D query time");
        validate(timing.d2h_scores_milliseconds, "D2H score time");
        validate(timing.cpu_topk_milliseconds, "CPU Top-K time");
        validate(timing.cleanup_milliseconds, "cleanup time");
        validate(timing.total_e2e_milliseconds, "E2E time");
    }
    if (timing.has_database_preparation_timing) {
        validate(
            timing.database_allocation_milliseconds,
            "database allocation time");
        validate(timing.database_h2d_milliseconds, "database H2D time");
        validate(
            timing.prepare_total_milliseconds,
            "database preparation time");
    }
    if (timing.has_query_stage_timing) {
        validate(
            timing.query_allocation_milliseconds,
            "query allocation time");
        validate(timing.query_h2d_milliseconds, "query H2D time");
        validate(
            timing.query_d2h_scores_milliseconds,
            "query D2H score time");
        validate(timing.query_cpu_topk_milliseconds, "query CPU Top-K time");
        validate(timing.query_cleanup_milliseconds, "query cleanup time");
        validate(timing.query_e2e_milliseconds, "query E2E time");
    }
}

}  // namespace

BenchmarkResult benchmark_search(
    const SearchBackend& backend,
    const SearchRequest& request,
    const BenchmarkConfig& config) {
    if (config.measured_iterations == 0) {
        throw std::invalid_argument("measured_iterations must be greater than zero");
    }
    const BackendTiming initial_timing = backend.last_timing();
    validate_timing(initial_timing);
    bool has_database_preparation_timing =
        initial_timing.has_database_preparation_timing;
    BackendTiming database_preparation_timing = initial_timing;


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
    std::size_t stage_timed_iterations = 0;
    double allocation_total_milliseconds = 0.0;
    double h2d_database_total_milliseconds = 0.0;
    double h2d_queries_total_milliseconds = 0.0;
    double d2h_scores_total_milliseconds = 0.0;
    double cpu_topk_total_milliseconds = 0.0;
    double cleanup_total_milliseconds = 0.0;
    double total_e2e_total_milliseconds = 0.0;

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
        validate_timing(timing);
        if (timing.has_kernel_latency) {
            has_kernel_timing = true;
            kernel_total_milliseconds += timing.kernel_latency_milliseconds;
            kernel_min_milliseconds = std::min(
                kernel_min_milliseconds, timing.kernel_latency_milliseconds);
            kernel_max_milliseconds = std::max(
                kernel_max_milliseconds, timing.kernel_latency_milliseconds);
        }

        if (timing.has_stage_timing) {
            ++stage_timed_iterations;
            allocation_total_milliseconds += timing.allocation_milliseconds;
            h2d_database_total_milliseconds +=
                timing.h2d_database_milliseconds;
            h2d_queries_total_milliseconds += timing.h2d_queries_milliseconds;
            d2h_scores_total_milliseconds += timing.d2h_scores_milliseconds;
            cpu_topk_total_milliseconds += timing.cpu_topk_milliseconds;
            cleanup_total_milliseconds += timing.cleanup_milliseconds;
            total_e2e_total_milliseconds += timing.total_e2e_milliseconds;
        }
        if (timing.has_database_preparation_timing &&
            !has_database_preparation_timing) {
            has_database_preparation_timing = true;
            database_preparation_timing = timing;
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
    if (stage_timed_iterations > 0) {
        const double stage_count =
            static_cast<double>(stage_timed_iterations);
        benchmark.has_stage_timing = true;
        benchmark.stage_timed_iterations = stage_timed_iterations;
        benchmark.allocation_average_milliseconds =
            allocation_total_milliseconds / stage_count;
        benchmark.h2d_database_average_milliseconds =
            h2d_database_total_milliseconds / stage_count;
        benchmark.h2d_queries_average_milliseconds =
            h2d_queries_total_milliseconds / stage_count;
        benchmark.d2h_scores_average_milliseconds =
            d2h_scores_total_milliseconds / stage_count;
        benchmark.cpu_topk_average_milliseconds =
            cpu_topk_total_milliseconds / stage_count;
        benchmark.cleanup_average_milliseconds =
            cleanup_total_milliseconds / stage_count;
        benchmark.total_e2e_average_milliseconds =
            total_e2e_total_milliseconds / stage_count;
    }
    if (has_database_preparation_timing) {
        benchmark.has_database_preparation_timing = true;
        benchmark.database_allocation_average_milliseconds =
            database_preparation_timing.database_allocation_milliseconds;
        benchmark.database_h2d_average_milliseconds =
            database_preparation_timing.database_h2d_milliseconds;
        benchmark.prepare_total_average_milliseconds =
            database_preparation_timing.prepare_total_milliseconds;
    }
    return benchmark;
}

RepeatedBenchmarkResult benchmark_repeated_search(
    const SearchBackend& backend,
    const std::vector<SearchRequest>& requests,
    const RepeatedBenchmarkConfig& config) {
    if (config.measured_batches == 0) {
        throw std::invalid_argument(
            "measured_batches must be greater than zero");
    }
    const std::size_t required_requests =
        config.warmup_batches + config.measured_batches;
    if (requests.size() != required_requests) {
        throw std::invalid_argument(
            "request count must equal warm-up plus measured batches");
    }
    const BackendTiming initial_timing = backend.last_timing();
    validate_timing(initial_timing);


    RepeatedBenchmarkResult benchmark;
    benchmark.warmup_batches = config.warmup_batches;
    benchmark.measured_batches = config.measured_batches;
    benchmark.measured_latencies_milliseconds.reserve(config.measured_batches);
    benchmark.measured_timings.reserve(config.measured_batches);
    if (initial_timing.has_database_preparation_timing) {
        benchmark.has_database_preparation_timing = true;
        benchmark.database_preparation_timing = initial_timing;
    }


    for (std::size_t index = 0; index < config.warmup_batches; ++index) {
        benchmark.result_checksum += checksum(backend.search(requests[index]));
    }

    for (std::size_t index = config.warmup_batches;
         index < required_requests;
         ++index) {
        const auto start = std::chrono::steady_clock::now();
        const SearchResult result = backend.search(requests[index]);
        const auto stop = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double, std::milli>(stop - start).count();
        if (!std::isfinite(elapsed) || elapsed < 0.0) {
            throw std::runtime_error("benchmark reported an invalid latency");
        }

        benchmark.measured_latencies_milliseconds.push_back(elapsed);
        const BackendTiming timing = backend.last_timing();
        validate_timing(timing);
        benchmark.measured_timings.push_back(timing);
        benchmark.result_checksum += checksum(result);
    }

    return benchmark;
}

}  // namespace vector_search
