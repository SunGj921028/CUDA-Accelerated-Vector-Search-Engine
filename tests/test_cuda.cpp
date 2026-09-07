#include "dataset.hpp"
#include "test_framework.hpp"
#include "vector_search.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace {

void require_results_match(
    const vector_search::SearchResult& expected,
    const vector_search::SearchResult& actual,
    float tolerance) {
    REQUIRE_EQ(actual.num_queries, expected.num_queries);
    REQUIRE_EQ(actual.topk, expected.topk);
    REQUIRE_EQ(actual.hits.size(), expected.hits.size());

    for (std::size_t query_index = 0;
         query_index < expected.num_queries;
         ++query_index) {
        for (std::size_t rank = 0; rank < expected.topk; ++rank) {
            const vector_search::SearchHit& expected_hit =
                expected.at(query_index, rank);
            const vector_search::SearchHit& actual_hit =
                actual.at(query_index, rank);
            REQUIRE_EQ(actual_hit.index, expected_hit.index);
            REQUIRE_NEAR(actual_hit.score, expected_hit.score, tolerance);
        }
    }
}

}  // namespace

TEST_CASE("cuda-naive matches CPU on normalized irregular shapes and Top-K") {
    std::vector<float> database{
        3.0F, 4.0F, 0.0F,
        1.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 1.0F,
        -1.0F, 0.0F, 0.0F,
    };
    std::vector<float> queries{
        3.0F, 4.0F, 0.0F,
        0.0F, 0.6F, 0.8F,
    };
    vector_search::normalize_vectors(database, 5, 3);
    vector_search::normalize_vectors(queries, 2, 3);

    const vector_search::SearchRequest request{
        database.data(), queries.data(), 5, 3, 2, 3};
    const std::unique_ptr<vector_search::SearchBackend> cpu =
        vector_search::create_backend("cpu");
    const std::unique_ptr<vector_search::SearchBackend> cuda =
        vector_search::create_backend("cuda-naive");

    const vector_search::SearchResult expected = cpu->search(request);
    const vector_search::SearchResult actual = cuda->search(request);
    require_results_match(expected, actual, 2.0e-5F);
    REQUIRE(cuda->last_timing().has_kernel_latency);
    REQUIRE(std::isfinite(cuda->last_timing().kernel_latency_milliseconds));
}

TEST_CASE("cuda-naive matches CPU for K=1 and known dot products") {
    const std::vector<float> database{
        1.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 1.0F,
        -1.0F, 0.0F, 0.0F,
        0.0F, -1.0F, 0.0F,
    };
    const std::vector<float> queries{
        0.8F, 0.6F, 0.0F,
        0.0F, 0.6F, 0.8F,
    };
    const vector_search::SearchRequest request{
        database.data(), queries.data(), 5, 3, 2, 1};

    const std::unique_ptr<vector_search::SearchBackend> cpu =
        vector_search::create_backend("cpu");
    const std::unique_ptr<vector_search::SearchBackend> cuda =
        vector_search::create_backend("cuda-naive");

    const vector_search::SearchResult expected = cpu->search(request);
    const vector_search::SearchResult actual = cuda->search(request);
    require_results_match(expected, actual, 2.0e-6F);

    REQUIRE_EQ(actual.at(0, 0).index, 0U);
    REQUIRE_NEAR(actual.at(0, 0).score, 0.8F, 2.0e-6F);
    REQUIRE_EQ(actual.at(1, 0).index, 2U);
    REQUIRE_NEAR(actual.at(1, 0).score, 0.8F, 2.0e-6F);
}

int main() {
    return vector_search_test::run_all();
}
