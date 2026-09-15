#include "cuda_backend.hpp"

#include "cuda_utils.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vector_search {
namespace {

// Resident mode deliberately uses the exact warp mapping and launch configuration.
// The controlled change is only the lifetime of the database allocation.
constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kWarpSize = 32;
constexpr unsigned int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
static_assert(kThreadsPerBlock % kWarpSize == 0);

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

void validate_database_request(const DatabaseRequest& request) {
    if (request.database == nullptr) {
        throw std::invalid_argument("database pointer must not be null");
    }
    if (request.num_vectors == 0) {
        throw std::invalid_argument("num_vectors must be greater than zero");
    }
    if (request.dimension == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }
}

void validate_query_request(
    const SearchRequest& request,
    const ResidentDatabaseInfo& database,
    const float* prepared_host_database) {
    if (request.queries == nullptr) {
        throw std::invalid_argument("query pointer must not be null");
    }
    if (request.database != nullptr &&
        request.database != prepared_host_database) {
        throw std::invalid_argument(
            "query database pointer does not match the prepared database; "
            "call prepare_database or reload_database");
    }
    if (request.num_vectors != database.num_vectors) {
        throw std::invalid_argument(
            "query num_vectors is incompatible with the prepared database");
    }
    if (request.dimension != database.dimension) {
        throw std::invalid_argument(
            "query dimension is incompatible with the prepared database");
    }
    if (request.num_queries == 0) {
        throw std::invalid_argument("num_queries must be greater than zero");
    }
    if (request.topk == 0 || request.topk > database.num_vectors) {
        throw std::invalid_argument(
            "topk must be greater than zero and no greater than num_vectors");
    }
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

double elapsed_host_milliseconds(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
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

float measure_synchronous_cuda_copy(
    CudaEvent& start,
    CudaEvent& stop,
    void* destination,
    const void* source,
    std::size_t bytes,
    cudaMemcpyKind kind) {
    start.record();
    VECTOR_SEARCH_CUDA_CHECK(cudaMemcpy(
        destination, source, bytes, kind));
    stop.record();
    stop.synchronize();
    const float milliseconds = stop.elapsed_since(start);
    if (!std::isfinite(milliseconds) || milliseconds < 0.0F) {
        throw std::runtime_error("CUDA event reported an invalid copy time");
    }
    return milliseconds;
}

// This is the same warp-per-vector kernel: 256 threads per block,
// eight warps per block, lane-strided dimensions, and shuffle-down reduction.
__global__ void warp_similarity_kernel(
    const float* database,
    const float* queries,
    float* scores,
    std::size_t num_vectors,
    std::size_t dimension,
    std::size_t total_pairs) {
    const unsigned int lane_id = threadIdx.x % warpSize;
    const unsigned int warp_id_in_block = threadIdx.x / warpSize;
    const std::size_t global_warp_id =
        static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock +
        warp_id_in_block;
    const std::size_t pair_index = global_warp_id;
    const bool pair_is_valid = pair_index < total_pairs;

    const unsigned int active_mask =
        __ballot_sync(0xffffffffU, pair_is_valid);
    if (!pair_is_valid) {
        return;
    }

    const std::size_t query_index = pair_index / num_vectors;
    const std::size_t vector_index = pair_index % num_vectors;
    const std::size_t query_base = query_index * dimension;
    const std::size_t database_base = vector_index * dimension;

    float partial = 0.0F;
    for (std::size_t dimension_index = lane_id;
         dimension_index < dimension;
         dimension_index += warpSize) {
        partial += queries[query_base + dimension_index] *
                   database[database_base + dimension_index];
    }

    for (int offset = static_cast<int>(warpSize / 2);
         offset > 0;
         offset >>= 1) {
        partial += __shfl_down_sync(active_mask, partial, offset);
    }

    if (lane_id == 0) {
        scores[pair_index] = partial;
    }
}

class CudaWarpResidentSearchBackend final : public ResidentSearchBackend {
public:
    std::string name() const override {
        return "cuda-warp-resident";
    }

    void prepare_database(const DatabaseRequest& request) override {
        prepare_database_impl(request, false);
    }

    void reload_database(const DatabaseRequest& request) override {
        prepare_database_impl(request, true);
    }

    void clear_database() override {
        device_database_.release();
        database_prepared_ = false;
        prepared_host_database_ = nullptr;
        database_info_ = {};
        last_timing_ = {};
    }

    bool database_is_prepared() const noexcept override {
        return database_prepared_;
    }

    ResidentDatabaseInfo database_info() const override {
        return database_info_;
    }

    SearchResult search(const SearchRequest& request) const override {
        if (!database_prepared_) {
            throw std::logic_error(
                "resident database is not initialized; call prepare_database");
        }
        validate_query_request(
            request, database_info_, prepared_host_database_);

        last_timing_ = {};

        const bool collect_stage_timing = timing_breakdown_enabled_;
        std::chrono::steady_clock::time_point total_start;
        if (collect_stage_timing) {
            total_start = std::chrono::steady_clock::now();
        }

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
            total_pairs / kWarpsPerBlock +
            (total_pairs % kWarpsPerBlock == 0 ? 0U : 1U);
        if (block_count >
            static_cast<std::size_t>(properties.maxGridSize[0])) {
            throw std::length_error(
                "CUDA workload requires more blocks than the device supports");
        }

        DeviceBuffer<float> device_queries;
        DeviceBuffer<float> device_scores;
        std::chrono::steady_clock::time_point allocation_start;
        if (collect_stage_timing) {
            allocation_start = std::chrono::steady_clock::now();
        }
        device_queries.allocate(query_elements);
        device_scores.allocate(total_pairs);
        double query_allocation_milliseconds = 0.0;
        if (collect_stage_timing) {
            query_allocation_milliseconds = elapsed_host_milliseconds(
                allocation_start, std::chrono::steady_clock::now());
        }

        CudaEvent stage_start;
        CudaEvent stage_stop;
        if (collect_stage_timing) {
            stage_start.create();
            stage_stop.create();
        }

        double query_h2d_milliseconds = 0.0;
        if (collect_stage_timing) {
            query_h2d_milliseconds = measure_synchronous_cuda_copy(
                stage_start,
                stage_stop,
                device_queries.get(),
                request.queries,
                checked_float_bytes(query_elements),
                cudaMemcpyHostToDevice);
        } else {
            VECTOR_SEARCH_CUDA_CHECK(cudaMemcpy(
                device_queries.get(),
                request.queries,
                checked_float_bytes(query_elements),
                cudaMemcpyHostToDevice));
        }

        CudaEvent kernel_start;
        CudaEvent kernel_stop;
        kernel_start.create();
        kernel_stop.create();
        kernel_start.record();

        warp_similarity_kernel<<<
            static_cast<unsigned int>(block_count), kThreadsPerBlock>>>(
            device_database_.get(),
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

        std::vector<float> host_scores(total_pairs, 0.0F);
        double query_d2h_scores_milliseconds = 0.0;
        if (collect_stage_timing) {
            query_d2h_scores_milliseconds = measure_synchronous_cuda_copy(
                stage_start,
                stage_stop,
                host_scores.data(),
                device_scores.get(),
                checked_float_bytes(total_pairs),
                cudaMemcpyDeviceToHost);
        } else {
            VECTOR_SEARCH_CUDA_CHECK(cudaMemcpy(
                host_scores.data(),
                device_scores.get(),
                checked_float_bytes(total_pairs),
                cudaMemcpyDeviceToHost));
        }

        SearchResult result;
        result.num_queries = request.num_queries;
        result.topk = request.topk;
        result.hits.reserve(result_elements);
        std::chrono::steady_clock::time_point cpu_topk_start;
        if (collect_stage_timing) {
            cpu_topk_start = std::chrono::steady_clock::now();
        }
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
        double query_cpu_topk_milliseconds = 0.0;
        if (collect_stage_timing) {
            query_cpu_topk_milliseconds = elapsed_host_milliseconds(
                cpu_topk_start, std::chrono::steady_clock::now());
        }

        std::chrono::steady_clock::time_point cleanup_start;
        if (collect_stage_timing) {
            cleanup_start = std::chrono::steady_clock::now();
            stage_stop.release();
            stage_start.release();
        }
        kernel_stop.release();
        kernel_start.release();
        device_scores.release();
        device_queries.release();

        BackendTiming timing = last_timing_;
        timing.has_kernel_latency = true;
        timing.kernel_latency_milliseconds = kernel_milliseconds;
        if (collect_stage_timing) {
            const auto cleanup_stop = std::chrono::steady_clock::now();
            const double query_cleanup_milliseconds = elapsed_host_milliseconds(
                cleanup_start, cleanup_stop);
            const double query_e2e_milliseconds = elapsed_host_milliseconds(
                total_start, cleanup_stop);

            timing.has_stage_timing = true;
            timing.allocation_milliseconds = query_allocation_milliseconds;
            timing.h2d_database_milliseconds = 0.0;
            timing.h2d_queries_milliseconds = query_h2d_milliseconds;
            timing.d2h_scores_milliseconds = query_d2h_scores_milliseconds;
            timing.cpu_topk_milliseconds = query_cpu_topk_milliseconds;
            timing.cleanup_milliseconds = query_cleanup_milliseconds;
            timing.total_e2e_milliseconds = query_e2e_milliseconds;

            timing.has_query_stage_timing = true;
            timing.query_allocation_milliseconds =
                query_allocation_milliseconds;
            timing.query_h2d_milliseconds = query_h2d_milliseconds;
            timing.query_d2h_scores_milliseconds =
                query_d2h_scores_milliseconds;
            timing.query_cpu_topk_milliseconds = query_cpu_topk_milliseconds;
            timing.query_cleanup_milliseconds = query_cleanup_milliseconds;
            timing.query_e2e_milliseconds = query_e2e_milliseconds;
        }
        last_timing_ = timing;
        return result;
    }

    BackendTiming last_timing() const override {
        return last_timing_;
    }

    void set_timing_breakdown_enabled(bool enabled) override {
        timing_breakdown_enabled_ = enabled;
    }

private:
    void prepare_database_impl(
        const DatabaseRequest& request,
        bool force_reload) {
        validate_database_request(request);
        if (!force_reload && database_prepared_ &&
            request.database == prepared_host_database_ &&
            request.num_vectors == database_info_.num_vectors &&
            request.dimension == database_info_.dimension) {
            return;
        }

        last_timing_ = {};
        const bool collect_stage_timing = timing_breakdown_enabled_;
        const auto prepare_start = std::chrono::steady_clock::now();

        std::size_t free_bytes_before = 0;
        std::size_t total_device_bytes = 0;
        VECTOR_SEARCH_CUDA_CHECK(cudaMemGetInfo(
            &free_bytes_before, &total_device_bytes));
        (void)total_device_bytes;

        database_prepared_ = false;
        prepared_host_database_ = nullptr;
        database_info_ = {};
        device_database_.release();

        const std::size_t database_elements = checked_product(
            request.num_vectors, request.dimension);
        const std::size_t database_bytes =
            checked_float_bytes(database_elements);
        const auto allocation_start = std::chrono::steady_clock::now();
        device_database_.allocate(database_elements);
        const double allocation_milliseconds = elapsed_host_milliseconds(
            allocation_start, std::chrono::steady_clock::now());

        CudaEvent stage_start;
        CudaEvent stage_stop;
        if (collect_stage_timing) {
            stage_start.create();
            stage_stop.create();
        }

        double h2d_milliseconds = 0.0;
        if (collect_stage_timing) {
            h2d_milliseconds = measure_synchronous_cuda_copy(
                stage_start,
                stage_stop,
                device_database_.get(),
                request.database,
                database_bytes,
                cudaMemcpyHostToDevice);
            stage_stop.release();
            stage_start.release();
        } else {
            VECTOR_SEARCH_CUDA_CHECK(cudaMemcpy(
                device_database_.get(),
                request.database,
                database_bytes,
                cudaMemcpyHostToDevice));
        }

        std::size_t free_bytes_after = 0;
        total_device_bytes = 0;
        VECTOR_SEARCH_CUDA_CHECK(cudaMemGetInfo(
            &free_bytes_after, &total_device_bytes));
        (void)total_device_bytes;

        database_prepared_ = true;
        prepared_host_database_ = request.database;
        database_info_.is_prepared = true;
        database_info_.num_vectors = request.num_vectors;
        database_info_.dimension = request.dimension;
        database_info_.bytes = database_bytes;
        database_info_.free_bytes_before_prepare = free_bytes_before;
        database_info_.free_bytes_after_prepare = free_bytes_after;

        if (collect_stage_timing) {
            last_timing_.has_database_preparation_timing = true;
            last_timing_.database_allocation_milliseconds =
                allocation_milliseconds;
            last_timing_.database_h2d_milliseconds = h2d_milliseconds;
            last_timing_.prepare_total_milliseconds = elapsed_host_milliseconds(
                prepare_start, std::chrono::steady_clock::now());
        }
    }

    DeviceBuffer<float> device_database_;
    bool database_prepared_ = false;
    const float* prepared_host_database_ = nullptr;
    ResidentDatabaseInfo database_info_{};
    mutable BackendTiming last_timing_{};
    mutable bool timing_breakdown_enabled_ = false;
};

}  // namespace

std::unique_ptr<ResidentSearchBackend> create_cuda_warp_resident_backend() {
    return std::make_unique<CudaWarpResidentSearchBackend>();
}

}  // namespace vector_search
