#include "vector_search.hpp"

#include "cuda_backend.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace vector_search {
namespace {

std::size_t checked_product(std::size_t left, std::size_t right) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error("search result dimensions overflow size_t");
    }
    return left * right;
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

class CpuSearchBackend final : public SearchBackend {
public:
    std::string name() const override {
        return "cpu";
    }

    SearchResult search(const SearchRequest& request) const override {
        validate_search_request(request);

        SearchResult result;
        result.num_queries = request.num_queries;
        result.topk = request.topk;
        result.hits.reserve(checked_product(request.num_queries, request.topk));

        for (std::size_t query_index = 0;
             query_index < request.num_queries;
             ++query_index) {
            const float* query = request.queries + query_index * request.dimension;
            const std::vector<float> scores = compute_similarity_scores(
                request.database,
                query,
                request.num_vectors,
                request.dimension);
            const std::vector<SearchHit> top_hits =
                select_top_k(scores, request.topk);
            result.hits.insert(result.hits.end(), top_hits.begin(), top_hits.end());
        }

        return result;
    }
};

}  // namespace

const SearchHit& SearchResult::at(
    std::size_t query_index,
    std::size_t rank) const {
    if (query_index >= num_queries || rank >= topk) {
        throw std::out_of_range("search result index out of range");
    }
    return hits.at(query_index * topk + rank);
}

std::vector<float> compute_similarity_scores(
    const float* database,
    const float* query,
    std::size_t num_vectors,
    std::size_t dimension) {
    if (database == nullptr || query == nullptr) {
        throw std::invalid_argument("similarity input pointers must not be null");
    }
    if (num_vectors == 0) {
        throw std::invalid_argument("num_vectors must be greater than zero");
    }
    if (dimension == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }

    std::vector<float> scores(num_vectors, 0.0F);
    for (std::size_t vector_index = 0;
         vector_index < num_vectors;
         ++vector_index) {
        const float* database_vector = database + vector_index * dimension;
        float score = 0.0F;
        for (std::size_t column = 0; column < dimension; ++column) {
            score += query[column] * database_vector[column];
        }
        scores[vector_index] = score;
    }
    return scores;
}

std::unique_ptr<SearchBackend> create_backend(const std::string& backend_name) {
    if (backend_name == "cpu") {
        return std::make_unique<CpuSearchBackend>();
    }

    if (backend_name == "cuda-naive") {
#if defined(VECTOR_SEARCH_ENABLE_CUDA)
        return create_cuda_naive_backend();
#else
        throw std::invalid_argument(
            "backend 'cuda-naive' is unavailable; configure with "
            "-DENABLE_CUDA=ON");
#endif
    }

    throw std::invalid_argument(
        "unknown backend '" + backend_name +
        "'; available backends are cpu and, when enabled, cuda-naive");
}

}  // namespace vector_search
