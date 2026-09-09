#include "benchmark.hpp"
#include "dataset.hpp"
#include "vector_search.hpp"

#if defined(VECTOR_SEARCH_ENABLE_CUDA)
#include "cuda_backend.hpp"
#endif

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

struct CommandLineOptions {
    vector_search::DatasetConfig dataset{10000, 128, 1, 42};
    std::size_t topk = 10;
    std::string backend = "cpu";
    bool benchmark = false;
    bool timing_breakdown = false;
    std::size_t repeat_batches = 0;
    vector_search::BenchmarkConfig benchmark_config{1, 5};
};

template <typename Integer>
Integer parse_integer(const std::string& option, const std::string& value) {
    if (value.empty()) {
        throw std::invalid_argument(
            option + " requires a numeric value");
    }
    if (value.front() == '-' || value.front() == '+') {
        throw std::invalid_argument(
            "invalid value for " + option + ": " + value);
    }

    unsigned long long parsed = 0;
    std::size_t consumed = 0;
    try {
        parsed = std::stoull(value, &consumed, 10);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "invalid value for " + option + ": " + value);
    }
    if (consumed != value.size() ||
        parsed > static_cast<unsigned long long>(
                     std::numeric_limits<Integer>::max())) {
        throw std::invalid_argument(
            "invalid value for " + option + ": " + value);
    }
    return static_cast<Integer>(parsed);
}

template <typename Integer>
Integer parse_positive_integer(
    const std::string& option,
    const std::string& value) {
    const Integer parsed = parse_integer<Integer>(option, value);
    if (parsed == 0) {
        throw std::invalid_argument(
            std::string(option) + " must be greater than zero");
    }
    return parsed;
}

std::string require_option_value(
    int& index,
    int argc,
    char* argv[],
    const std::string& option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            std::string(option) + " requires a value");
    }
    ++index;
    return argv[index];
}

CommandLineOptions parse_command_line(int argc, char* argv[]) {
    CommandLineOptions options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            std::cout <<
                "Usage: vector_search [options]\n"
                "  --backend <name>       Search backend (default: cpu)\n"
                "  --vectors <count>      Database vector count\n"
                "  --dim <count>          Vector dimension\n"
                "  --queries <count>      Query vector count\n"
                "  --topk <count>         Number of results per query\n"
                "  --seed <value>         Synthetic data seed\n"
                "  --benchmark            Run timed search iterations\n"
                "  --timing-breakdown     Report detailed CUDA stage timing\n"
                "  --repeat-batches <n>   Run sequential query-batch benchmark\n"
                "  --warmup <count>       Benchmark warm-up iterations\n"
                "  --iterations <count>   Measured benchmark iterations\n"
                "  --help                 Show this help text\n";
            options.benchmark = false;
            options.backend = "__help__";
            continue;
        }
        if (argument == "--benchmark") {
            options.benchmark = true;
            continue;
        }
        if (argument == "--timing-breakdown") {
            options.timing_breakdown = true;
            continue;
        }
        if (argument == "--repeat-batches") {
            options.repeat_batches = parse_positive_integer<std::size_t>(
                argument, require_option_value(index, argc, argv, argument));
            options.benchmark = true;
            continue;
        }
        if (argument == "--backend") {
            options.backend = require_option_value(index, argc, argv, argument);
            continue;
        }
        if (argument == "--vectors") {
            options.dataset.num_vectors = parse_positive_integer<std::size_t>(
                argument, require_option_value(index, argc, argv, argument));
            continue;
        }
        if (argument == "--dim") {
            options.dataset.dimension = parse_positive_integer<std::size_t>(
                argument, require_option_value(index, argc, argv, argument));
            continue;
        }
        if (argument == "--queries") {
            options.dataset.num_queries = parse_positive_integer<std::size_t>(
                argument, require_option_value(index, argc, argv, argument));
            continue;
        }
        if (argument == "--topk") {
            options.topk = parse_positive_integer<std::size_t>(
                argument, require_option_value(index, argc, argv, argument));
            continue;
        }
        if (argument == "--seed") {
            options.dataset.seed = parse_integer<std::uint64_t>(
                argument, require_option_value(index, argc, argv, argument));
            continue;
        }
        if (argument == "--warmup") {
            options.benchmark_config.warmup_iterations =
                parse_integer<std::size_t>(
                    argument, require_option_value(index, argc, argv, argument));
            continue;
        }
        if (argument == "--iterations") {
            options.benchmark_config.measured_iterations =
                parse_positive_integer<std::size_t>(
                    argument, require_option_value(index, argc, argv, argument));
            continue;
        }

        throw std::invalid_argument(
            "unknown command-line option: " + std::string(argument));
    }

    return options;
}

std::string compiler_description() {
#if defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    return "Clang " + std::to_string(__clang_major__) + "." +
           std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." +
           std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#else
    return "unknown";
#endif
}

std::string logical_processor_description() {
    const char* processor_count = std::getenv("NUMBER_OF_PROCESSORS");
    if (processor_count != nullptr) {
        return processor_count;
    }

    const unsigned int detected_processors = std::thread::hardware_concurrency();
    return detected_processors == 0
               ? "unknown"
               : std::to_string(detected_processors);
}

void print_database_info(const vector_search::ResidentDatabaseInfo& info) {
    if (!info.is_prepared) {
        return;
    }
    std::cout << std::fixed << std::setprecision(3)
              << "database_bytes=" << info.bytes << '\n'
              << "database_mib="
              << static_cast<double>(info.bytes) / (1024.0 * 1024.0) << '\n'
              << "device_free_bytes_before_prepare="
              << info.free_bytes_before_prepare << '\n'
              << "device_free_bytes_after_prepare="
              << info.free_bytes_after_prepare << '\n';
}

std::size_t checked_product(std::size_t left, std::size_t right) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error("benchmark dimensions overflow size_t");
    }
    return left * right;
}

std::size_t checked_sum(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::length_error("benchmark batch count overflows size_t");
    }
    return left + right;
}

void print_configuration(
    const CommandLineOptions& options,
    const vector_search::SearchBackend& backend,
    std::size_t generated_query_count) {
    std::cout << "backend=" << backend.name() << '\n'
              << "cpu_logical_processors="
              << logical_processor_description() << '\n';

#if defined(VECTOR_SEARCH_ENABLE_CUDA)
    const vector_search::CudaRuntimeInfo cuda_info =
        vector_search::query_cuda_runtime_info();
    if (cuda_info.available) {
        std::cout << "gpu=" << cuda_info.device_name << '\n'
                  << "cuda_toolkit=" << cuda_info.toolkit_version << '\n';
    } else {
        std::cout << "gpu=unavailable (" << cuda_info.error << ")\n"
                  << "cuda_toolkit=unavailable\n";
    }
#else
    std::cout << "gpu=unavailable (CUDA not enabled)\n"
              << "cuda_toolkit=not built\n";
#endif

    std::cout << "compiler=" << compiler_description() << '\n'
              << "vectors=" << options.dataset.num_vectors << '\n'
              << "dim=" << options.dataset.dimension << '\n'
              << "queries=" << options.dataset.num_queries << '\n'
              << "topk=" << options.topk << '\n'
              << "seed=" << options.dataset.seed << '\n';
    if (options.repeat_batches > 0) {
        std::cout << "queries_per_batch=" << options.dataset.num_queries << '\n'
                  << "query_batches_total="
                  << checked_sum(
                         options.benchmark_config.warmup_iterations,
                         options.repeat_batches)
                  << '\n'
                  << "generated_queries=" << generated_query_count << '\n';
    }
}

int run(int argc, char* argv[]) {
    const CommandLineOptions options = parse_command_line(argc, argv);
    if (options.backend == "__help__") {
        return 0;
    }

    if (options.topk > options.dataset.num_vectors) {
        throw std::invalid_argument("--topk must not exceed --vectors");
    }
    vector_search::DatasetConfig dataset_config = options.dataset;
    std::size_t generated_query_count = options.dataset.num_queries;
    if (options.repeat_batches > 0) {
        const std::size_t total_batches = checked_sum(
            options.benchmark_config.warmup_iterations,
            options.repeat_batches);
        generated_query_count = checked_product(
            options.dataset.num_queries, total_batches);
        dataset_config.num_queries = generated_query_count;
    }


    const vector_search::Dataset dataset =
        vector_search::generate_synthetic_dataset(dataset_config);
    const vector_search::SearchRequest request{
        dataset.database.data(),
        dataset.queries.data(),
        dataset.num_vectors,
        dataset.dimension,
        dataset.num_queries,
        options.topk,
    };
    const std::unique_ptr<vector_search::SearchBackend> backend =
        vector_search::create_backend(options.backend);
    backend->set_timing_breakdown_enabled(options.timing_breakdown);
    vector_search::ResidentSearchBackend* resident_backend =
        dynamic_cast<vector_search::ResidentSearchBackend*>(backend.get());
    if (resident_backend != nullptr) {
        resident_backend->prepare_database({
            dataset.database.data(), dataset.num_vectors, dataset.dimension});
    }

    print_configuration(options, *backend, generated_query_count);


    if (options.benchmark) {
        if (options.repeat_batches > 0) {
            const std::size_t total_batches = checked_sum(
                options.benchmark_config.warmup_iterations,
                options.repeat_batches);
            const std::size_t query_elements_per_batch = checked_product(
                options.dataset.num_queries, dataset.dimension);
            std::vector<vector_search::SearchRequest> batch_requests;
            batch_requests.reserve(total_batches);
            for (std::size_t batch = 0; batch < total_batches; ++batch) {
                const std::size_t query_offset = checked_product(
                    batch, query_elements_per_batch);
                batch_requests.push_back({
                    dataset.database.data(),
                    dataset.queries.data() + query_offset,
                    dataset.num_vectors,
                    dataset.dimension,
                    options.dataset.num_queries,
                    options.topk});
            }

            const vector_search::RepeatedBenchmarkResult result =
                vector_search::benchmark_repeated_search(
                    *backend,
                    batch_requests,
                    {options.benchmark_config.warmup_iterations,
                     options.repeat_batches});
            std::cout << std::fixed << std::setprecision(6)
                      << "mode=repeated_benchmark\n"
                      << "timing_scope=sequential_search_batches_dataset_generation_excluded\n"
                      << "warmup_batches=" << result.warmup_batches << '\n'
                      << "measured_batches=" << result.measured_batches << '\n'
                      << "result_checksum=" << result.result_checksum << '\n';
            if (resident_backend != nullptr) {
                print_database_info(resident_backend->database_info());
            }
            if (result.has_database_preparation_timing) {
                const vector_search::BackendTiming& timing =
                    result.database_preparation_timing;
                std::cout << "db_allocation_ms="
                          << timing.database_allocation_milliseconds << '\n'
                          << "db_h2d_ms=" << timing.database_h2d_milliseconds
                          << '\n'
                          << "prepare_total_ms=" << timing.prepare_total_milliseconds
                          << '\n';
            }
            for (std::size_t index = 0;
                 index < result.measured_latencies_milliseconds.size();
                 ++index) {
                const std::size_t batch_number = index + 1;
                const vector_search::BackendTiming& timing =
                    result.measured_timings[index];
                std::cout << "batch_" << batch_number << "_latency_ms="
                          << result.measured_latencies_milliseconds[index]
                          << '\n';
                if (timing.has_kernel_latency) {
                    std::cout << "batch_" << batch_number << "_kernel_ms="
                              << timing.kernel_latency_milliseconds << '\n';
                }
                if (timing.has_query_stage_timing) {
                    std::cout << "batch_" << batch_number
                              << "_query_allocation_ms="
                              << timing.query_allocation_milliseconds << '\n'
                              << "batch_" << batch_number << "_query_h2d_ms="
                              << timing.query_h2d_milliseconds << '\n'
                              << "batch_" << batch_number
                              << "_query_d2h_scores_ms="
                              << timing.query_d2h_scores_milliseconds << '\n'
                              << "batch_" << batch_number
                              << "_query_cpu_topk_ms="
                              << timing.query_cpu_topk_milliseconds << '\n'
                              << "batch_" << batch_number
                              << "_query_cleanup_ms="
                              << timing.query_cleanup_milliseconds << '\n'
                              << "batch_" << batch_number << "_query_e2e_ms="
                              << timing.query_e2e_milliseconds << '\n';
                } else if (timing.has_stage_timing) {
                    std::cout << "batch_" << batch_number << "_allocation_ms="
                              << timing.allocation_milliseconds << '\n'
                              << "batch_" << batch_number
                              << "_h2d_database_ms="
                              << timing.h2d_database_milliseconds << '\n'
                              << "batch_" << batch_number
                              << "_h2d_queries_ms="
                              << timing.h2d_queries_milliseconds << '\n'
                              << "batch_" << batch_number << "_d2h_scores_ms="
                              << timing.d2h_scores_milliseconds << '\n'
                              << "batch_" << batch_number << "_cpu_topk_ms="
                              << timing.cpu_topk_milliseconds << '\n'
                              << "batch_" << batch_number << "_cleanup_ms="
                              << timing.cleanup_milliseconds << '\n'
                              << "batch_" << batch_number << "_e2e_ms="
                              << timing.total_e2e_milliseconds << '\n';
                }
            }
            return 0;
        }
        const vector_search::BenchmarkResult result =
            vector_search::benchmark_search(
                *backend, request, options.benchmark_config);
        std::cout << std::fixed << std::setprecision(3)
                  << "timing_scope=backend_search_dataset_generation_excluded\n"
                  << "warmup_iterations=" << result.warmup_iterations << '\n'
                  << "measured_iterations=" << result.measured_iterations << '\n'
                  << "latency_ms_min=" << result.min_milliseconds << '\n'
                  << "latency_ms_average=" << result.average_milliseconds << '\n'
                  << "latency_ms_max=" << result.max_milliseconds << '\n'
                  << "queries_per_second=" << result.queries_per_second << '\n'
                  << "result_checksum=" << result.result_checksum << '\n';
        if (result.has_kernel_timing) {
            std::cout << "kernel_timing_scope=cuda_similarity_kernel_only_cuda_events\n"
                      << "kernel_latency_ms_min="
                      << result.kernel_min_milliseconds << '\n'
                      << "kernel_latency_ms_average="
                      << result.kernel_average_milliseconds << '\n'
                      << "kernel_latency_ms_max="
                      << result.kernel_max_milliseconds << '\n'
                      << "end_to_end_timing_scope=allocation_h2d_kernel_d2h_cpu_topk\n"
                      << "end_to_end_latency_ms_average="
                      << result.average_milliseconds << '\n';
        }
        if (result.has_stage_timing) {
            std::cout << "stage_timing_scope=cuda_events_for_h2d_kernel_d2h_host_steady_clock_for_allocation_cpu_topk_cleanup_total\n"
                      << "stage_timed_iterations="
                      << result.stage_timed_iterations << '\n'
                      << "allocation_ms_average="
                      << result.allocation_average_milliseconds << '\n'
                      << "h2d_database_ms_average="
                      << result.h2d_database_average_milliseconds << '\n'
                      << "h2d_queries_ms_average="
                      << result.h2d_queries_average_milliseconds << '\n'
                      << "kernel_ms_average="
                      << result.kernel_average_milliseconds << '\n'
                      << "d2h_scores_ms_average="
                      << result.d2h_scores_average_milliseconds << '\n'
                      << "cpu_topk_ms_average="
                      << result.cpu_topk_average_milliseconds << '\n'
                      << "cleanup_ms_average="
                      << result.cleanup_average_milliseconds << '\n'
                      << "total_e2e_ms_average="
                      << result.total_e2e_average_milliseconds << '\n';
        }
        if (result.has_database_preparation_timing) {
            std::cout << "db_allocation_ms_average="
                      << result.database_allocation_average_milliseconds
                      << '\n'
                      << "db_h2d_ms_average="
                      << result.database_h2d_average_milliseconds << '\n'
                      << "prepare_total_ms_average="
                      << result.prepare_total_average_milliseconds << '\n';
            if (options.benchmark_config.warmup_iterations == 0 &&
                options.benchmark_config.measured_iterations == 1) {
                std::cout << "cold_start_prepare_plus_first_query_ms="
                          << result.prepare_total_average_milliseconds +
                                 result.average_milliseconds
                          << '\n';
            }
        }
        if (resident_backend != nullptr) {
            print_database_info(resident_backend->database_info());
        }
        return 0;
    }

    const vector_search::BackendTiming preparation_timing =
        resident_backend != nullptr ? backend->last_timing()
                                     : vector_search::BackendTiming{};
    const vector_search::SearchResult result = backend->search(request);
    std::cout << "mode=search\n"
              << "showing_query=0\n";
    for (std::size_t rank = 0; rank < result.topk; ++rank) {
        const vector_search::SearchHit& hit = result.at(0, rank);
        std::cout << "rank=" << rank
                  << " index=" << hit.index
                  << " score=" << std::setprecision(9) << hit.score << '\n';
    }
    if (options.timing_breakdown) {
        const vector_search::BackendTiming timing = backend->last_timing();
        std::cout << std::fixed << std::setprecision(3);
        if (preparation_timing.has_database_preparation_timing) {
            std::cout << "db_allocation_ms="
                      << preparation_timing.database_allocation_milliseconds
                      << '\n'
                      << "db_h2d_ms="
                      << preparation_timing.database_h2d_milliseconds << '\n'
                      << "prepare_total_ms="
                      << preparation_timing.prepare_total_milliseconds << '\n';
        }
        if (timing.has_query_stage_timing) {
            std::cout << "query_stage_timing_scope=cuda_events_for_query_h2d_kernel_query_d2h_host_steady_clock_for_query_allocation_cpu_topk_cleanup_total\n"
                      << "query_allocation_ms="
                      << timing.query_allocation_milliseconds << '\n'
                      << "query_h2d_ms=" << timing.query_h2d_milliseconds << '\n'
                      << "kernel_ms=" << timing.kernel_latency_milliseconds << '\n'
                      << "query_d2h_scores_ms="
                      << timing.query_d2h_scores_milliseconds << '\n'
                      << "query_cpu_topk_ms="
                      << timing.query_cpu_topk_milliseconds << '\n'
                      << "query_cleanup_ms="
                      << timing.query_cleanup_milliseconds << '\n'
                      << "query_e2e_ms=" << timing.query_e2e_milliseconds << '\n';
        } else if (!timing.has_stage_timing) {
            std::cout << "stage_timing=unavailable_for_backend\n";
        } else {
            std::cout
                << "stage_timing_scope=cuda_events_for_h2d_kernel_d2h_host_steady_clock_for_allocation_cpu_topk_cleanup_total\n"
                << "allocation_ms=" << timing.allocation_milliseconds << '\n'
                << "h2d_database_ms="
                << timing.h2d_database_milliseconds << '\n'
                << "h2d_queries_ms=" << timing.h2d_queries_milliseconds << '\n'
                << "kernel_ms=" << timing.kernel_latency_milliseconds << '\n'
                << "d2h_scores_ms=" << timing.d2h_scores_milliseconds << '\n'
                << "cpu_topk_ms=" << timing.cpu_topk_milliseconds << '\n'
                << "cleanup_ms=" << timing.cleanup_milliseconds << '\n'
                << "total_e2e_ms=" << timing.total_e2e_milliseconds << '\n';
        }
        if (resident_backend != nullptr) {
            print_database_info(resident_backend->database_info());
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n'
                  << "Use --help for usage.\n";
        return 1;
    }
}
