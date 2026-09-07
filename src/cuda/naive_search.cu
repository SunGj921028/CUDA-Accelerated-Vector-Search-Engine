#include "cuda_backend.hpp"

#include "cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vector_search {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;

std::size_t checked_product(std::size_t left, std::size_t right) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error("CUDA workload dimensions overflow size_t");
    }
    return left * right;
}

std::size_t checked_float_bytes(std::size_t element_count) {
    if (element_count >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::length_error("CUDA buffer size overflows size_t");
    }
    return element_count * sizeof(float);
}

void validate_search_request(const SearchRequest& request) {
    if (request.database == nullptr || request.queries == nullptr) {
        throw std::invalid_argument("search input pointers must not be null");
    }
    if (request.num_vectors == 0) {
        throw std::invalid_argument("num_vectors must be greater than zero");
    }
    if (request.dimension == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }
    if (request.num_queries == 0) {
        throw std::invalid_argument("num_queries must be greater than zero");
    }
    if (request.topk == 0 || request.topk > request.num_vectors) {
        throw std::invalid_argument(
            "topk must be greater than zero and no greater than num_vectors");
    }
}

std::string format_cuda_version(int version) {
    std::ostringstream formatted;
    formatted << version / 1000 << '.' << (version % 1000) / 10;
    return formatted.str();
}

void report_cleanup_error(
    cudaError_t status,
    const char* operation) noexcept {
    if (status != cudaSuccess) {
        std::cerr << vector_search::cuda_detail::format_cuda_error(
                         status, operation)
                  << '\n';
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            report_cleanup_error(cudaFree(pointer_), "cudaFree");
            pointer_ = nullptr;
        }
    }

    void allocate(std::size_t element_count) {
        if (element_count >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::length_error("CUDA buffer size overflows size_t");
        }
        const std::size_t bytes = element_count * sizeof(T);
        VECTOR_SEARCH_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&pointer_), bytes));
    }

    T* get() const {
        return pointer_;
    }

    void release() {
        if (pointer_ == nullptr) {
            return;
        }
        VECTOR_SEARCH_CUDA_CHECK(cudaFree(pointer_));
        pointer_ = nullptr;
    }

private:
    T* pointer_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() = default;
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    ~CudaEvent() {
        if (event_ != nullptr) {
            report_cleanup_error(cudaEventDestroy(event_), "cudaEventDestroy");
            event_ = nullptr;
        }
    }

    void create() {
        VECTOR_SEARCH_CUDA_CHECK(cudaEventCreate(&event_));
    }

    void record() {
        VECTOR_SEARCH_CUDA_CHECK(cudaEventRecord(event_, 0));
    }

    void synchronize() {
        VECTOR_SEARCH_CUDA_CHECK(cudaEventSynchronize(event_));
    }

    float elapsed_since(const CudaEvent& start) const {
        float milliseconds = 0.0F;
        VECTOR_SEARCH_CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds, start.event_, event_));
        return milliseconds;
    }

    void release() {
        if (event_ == nullptr) {
            return;
        }
        VECTOR_SEARCH_CUDA_CHECK(cudaEventDestroy(event_));
        event_ = nullptr;
    }

private:
    cudaEvent_t event_ = nullptr;
};

__global__ void naive_similarity_kernel(
    const float* database,
    const float* queries,
    float* scores,
    std::size_t num_vectors,
    std::size_t dimension,
    std::size_t total_pairs) {
    const std::size_t pair_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_index >= total_pairs) {
        return;
    }

    const std::size_t query_index = pair_index / num_vectors;
    const std::size_t vector_index = pair_index % num_vectors;
    const float* query = queries + query_index * dimension;
    const float* database_vector = database + vector_index * dimension;

    float score = 0.0F;
    for (std::size_t column = 0; column < dimension; ++column) {
        score += query[column] * database_vector[column];
    }
    scores[pair_index] = score;
}

class CudaNaiveSearchBackend final : public SearchBackend {
public:
    std::string name() const override {
        return "cuda-naive";
    }

    SearchResult search(const SearchRequest& request) const override {
        validate_search_request(request);
        last_timing_ = {};

        const std::size_t database_elements = checked_product(
            request.num_vectors, request.dimension);
        const std::size_t query_elements = checked_product(
            request.num_queries, request.dimension);
        const std::size_t total_pairs = checked_product(
            request.num_queries, request.num_vectors);
        const std::size_t result_elements = checked_product(
            request.num_queries, request.topk);

        int device = 0;
        VECTOR_SEARCH_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        VECTOR_SEARCH_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        const std::size_t block_count =
            total_pairs / kThreadsPerBlock +
            (total_pairs % kThreadsPerBlock == 0 ? 0 : 1);
        if (block_count > static_cast<std::size_t>(properties.maxGridSize[0])) {
            throw std::length_error(
                "CUDA workload requires more blocks than the device supports");
        }

        DeviceBuffer<float> device_database;
        DeviceBuffer<float> device_queries;
        DeviceBuffer<float> device_scores;
        device_database.allocate(database_elements);
        device_queries.allocate(query_elements);
        device_scores.allocate(total_pairs);

        VECTOR_SEARCH_CUDA_CHECK(cudaMemcpy(
            device_database.get(),
            request.database,
            checked_float_bytes(database_elements),
            cudaMemcpyHostToDevice));
        VECTOR_SEARCH_CUDA_CHECK(cudaMemcpy(
            device_queries.get(),
            request.queries,
            checked_float_bytes(query_elements),
            cudaMemcpyHostToDevice));

        CudaEvent kernel_start;
        CudaEvent kernel_stop;
        kernel_start.create();
        kernel_stop.create();
        kernel_start.record();

        naive_similarity_kernel<<<
            static_cast<unsigned int>(block_count), kThreadsPerBlock>>>(
            device_database.get(),
            device_queries.get(),
            device_scores.get(),
            request.num_vectors,
            request.dimension,
            total_pairs);
        VECTOR_SEARCH_CUDA_CHECK(cudaGetLastError());

        kernel_stop.record();
        kernel_stop.synchronize();
        const float kernel_milliseconds =
            kernel_stop.elapsed_since(kernel_start);
        if (!std::isfinite(kernel_milliseconds) || kernel_milliseconds < 0.0F) {
            throw std::runtime_error("CUDA event reported an invalid kernel time");
        }
        last_timing_.has_kernel_latency = true;
        last_timing_.kernel_latency_milliseconds = kernel_milliseconds;

        std::vector<float> host_scores(total_pairs, 0.0F);
        VECTOR_SEARCH_CUDA_CHECK(cudaMemcpy(
            host_scores.data(),
            device_scores.get(),
            checked_float_bytes(total_pairs),
            cudaMemcpyDeviceToHost));

        SearchResult result;
        result.num_queries = request.num_queries;
        result.topk = request.topk;
        result.hits.reserve(result_elements);
        for (std::size_t query_index = 0;
             query_index < request.num_queries;
             ++query_index) {
            const auto query_begin = host_scores.begin() +
                                     query_index * request.num_vectors;
            const std::vector<float> query_scores(
                query_begin, query_begin + request.num_vectors);
            const std::vector<SearchHit> top_hits =
                select_top_k(query_scores, request.topk);
            result.hits.insert(result.hits.end(), top_hits.begin(), top_hits.end());
        }

        kernel_stop.release();
        kernel_start.release();
        device_scores.release();
        device_queries.release();
        device_database.release();
        return result;
    }

    BackendTiming last_timing() const override {
        return last_timing_;
    }

private:
    mutable BackendTiming last_timing_{};
};

}  // namespace

CudaRuntimeInfo query_cuda_runtime_info() {
    CudaRuntimeInfo info;

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        info.error = cuda_detail::format_cuda_error(
            count_status, "cudaGetDeviceCount");
        return info;
    }
    if (device_count == 0) {
        info.error = "no CUDA device was reported";
        return info;
    }

    int device = 0;
    const cudaError_t device_status = cudaGetDevice(&device);
    if (device_status != cudaSuccess) {
        info.error = cuda_detail::format_cuda_error(
            device_status, "cudaGetDevice");
        return info;
    }

    cudaDeviceProp properties{};
    const cudaError_t properties_status =
        cudaGetDeviceProperties(&properties, device);
    if (properties_status != cudaSuccess) {
        info.error = cuda_detail::format_cuda_error(
            properties_status, "cudaGetDeviceProperties");
        return info;
    }

    info.available = true;
    info.device_name = properties.name;
    info.toolkit_version = format_cuda_version(CUDART_VERSION);
    return info;
}

std::unique_ptr<SearchBackend> create_cuda_naive_backend() {
    return std::make_unique<CudaNaiveSearchBackend>();
}

}  // namespace vector_search
