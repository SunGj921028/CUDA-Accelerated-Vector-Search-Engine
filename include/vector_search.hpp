#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vector_search {

struct DatabaseRequest {
    const float* database = nullptr;
    std::size_t num_vectors = 0;
    std::size_t dimension = 0;
};

struct SearchRequest {
    const float* database = nullptr;
    const float* queries = nullptr;
    std::size_t num_vectors = 0;
    std::size_t dimension = 0;
    std::size_t num_queries = 0;
    std::size_t topk = 0;
};

struct SearchHit {
    std::size_t index = 0;
    float score = 0.0F;
};

struct SearchResult {
    std::size_t num_queries = 0;
    std::size_t topk = 0;
    // Results are stored query-major: query 0's Top-K, then query 1's Top-K.
    std::vector<SearchHit> hits;

    const SearchHit& at(std::size_t query_index, std::size_t rank) const;
};

struct ResidentDatabaseInfo {
    bool is_prepared = false;
    std::size_t num_vectors = 0;
    std::size_t dimension = 0;
    std::size_t bytes = 0;
    std::size_t free_bytes_before_prepare = 0;
    std::size_t free_bytes_after_prepare = 0;
};

struct BackendTiming {
    bool has_kernel_latency = false;
    double kernel_latency_milliseconds = 0.0;

    // Detailed stage timing is opt-in so ordinary M1 benchmark runs retain
    // their original measurement behavior. CUDA backends populate these
    // fields when set_timing_breakdown_enabled(true) is requested.
    bool has_stage_timing = false;
    double allocation_milliseconds = 0.0;
    double h2d_database_milliseconds = 0.0;
    double h2d_queries_milliseconds = 0.0;
    double d2h_scores_milliseconds = 0.0;
    double cpu_topk_milliseconds = 0.0;
    double cleanup_milliseconds = 0.0;
    double total_e2e_milliseconds = 0.0;

    // M5 separates the one-time resident database preparation from the
    // per-query pipeline. These fields are populated by resident backends
    // when detailed timing is enabled.
    bool has_database_preparation_timing = false;
    double database_allocation_milliseconds = 0.0;
    double database_h2d_milliseconds = 0.0;
    double prepare_total_milliseconds = 0.0;

    bool has_query_stage_timing = false;
    double query_allocation_milliseconds = 0.0;
    double query_h2d_milliseconds = 0.0;
    double query_d2h_scores_milliseconds = 0.0;
    double query_cpu_topk_milliseconds = 0.0;
    double query_cleanup_milliseconds = 0.0;
    double query_e2e_milliseconds = 0.0;
};

class SearchBackend {
public:
    virtual ~SearchBackend() = default;

    virtual std::string name() const = 0;
    virtual SearchResult search(const SearchRequest& request) const = 0;

    // Backends with a device-specific kernel timer can expose the timing for
    // the most recent search. CPU and future backends can use the default.
    virtual BackendTiming last_timing() const {
        return {};
    }

    // Detailed stage instrumentation is disabled by default to avoid adding
    // synchronization and event overhead to baseline measurements.
    virtual void set_timing_breakdown_enabled(bool enabled) {
        (void)enabled;
    }
};

// A resident backend separates database lifetime from query lifetime. The
// inherited search request may leave database null after preparation; when it
// is non-null, a resident backend verifies that it identifies the prepared
// host database rather than silently using a different database.
class ResidentSearchBackend : public SearchBackend {
public:
    ~ResidentSearchBackend() override = default;

    virtual void prepare_database(const DatabaseRequest& request) = 0;
    virtual void reload_database(const DatabaseRequest& request) = 0;
    virtual void clear_database() = 0;
    virtual bool database_is_prepared() const noexcept = 0;
    virtual ResidentDatabaseInfo database_info() const = 0;
};

std::vector<float> compute_similarity_scores(
    const float* database,
    const float* query,
    std::size_t num_vectors,
    std::size_t dimension);

std::vector<SearchHit> select_top_k(
    const std::vector<float>& scores,
    std::size_t k);

std::unique_ptr<SearchBackend> create_backend(const std::string& backend_name);

}  // namespace vector_search
