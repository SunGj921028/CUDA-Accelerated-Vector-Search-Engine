#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vector_search {

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

class SearchBackend {
public:
    virtual ~SearchBackend() = default;

    virtual std::string name() const = 0;
    virtual SearchResult search(const SearchRequest& request) const = 0;
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
