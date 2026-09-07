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

void print_configuration(
    const CommandLineOptions& options,
    const vector_search::SearchBackend& backend) {
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
}

int run(int argc, char* argv[]) {
    const CommandLineOptions options = parse_command_line(argc, argv);
    if (options.backend == "__help__") {
        return 0;
    }

    if (options.topk > options.dataset.num_vectors) {
        throw std::invalid_argument("--topk must not exceed --vectors");
    }

    const vector_search::Dataset dataset =
        vector_search::generate_synthetic_dataset(options.dataset);
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

    print_configuration(options, *backend);

    if (options.benchmark) {
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
        return 0;
    }

    const vector_search::SearchResult result = backend->search(request);
    std::cout << "mode=search\n"
              << "showing_query=0\n";
    for (std::size_t rank = 0; rank < result.topk; ++rank) {
        const vector_search::SearchHit& hit = result.at(0, rank);
        std::cout << "rank=" << rank
                  << " index=" << hit.index
                  << " score=" << std::setprecision(9) << hit.score << '\n';
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
