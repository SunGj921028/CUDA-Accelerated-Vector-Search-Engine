#include "test_framework.hpp"
#include "vector_search.hpp"

#include <string>
#include <vector>

TEST_CASE("CPU backend returns query-major exact Top-K results") {
    const std::vector<float> database{
        1.0F, 0.0F,
        0.0F, 1.0F,
        -1.0F, 0.0F,
        0.0F, -1.0F,
    };
    const std::vector<float> queries{
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const vector_search::SearchRequest request{
        database.data(), queries.data(), 4, 2, 2, 2};

    const std::unique_ptr<vector_search::SearchBackend> backend =
        vector_search::create_backend("cpu");
    const vector_search::SearchResult result = backend->search(request);

    REQUIRE_EQ(result.num_queries, 2U);
    REQUIRE_EQ(result.topk, 2U);
    REQUIRE_EQ(result.hits.size(), 4U);

    REQUIRE_EQ(result.at(0, 0).index, 0U);
    REQUIRE_EQ(result.at(0, 1).index, 1U);
    REQUIRE_NEAR(result.at(0, 0).score, 1.0F, 1.0e-6F);
    REQUIRE_NEAR(result.at(0, 1).score, 0.0F, 1.0e-6F);

    REQUIRE_EQ(result.at(1, 0).index, 1U);
    REQUIRE_EQ(result.at(1, 1).index, 0U);
    REQUIRE_NEAR(result.at(1, 0).score, 1.0F, 1.0e-6F);
    REQUIRE_NEAR(result.at(1, 1).score, 0.0F, 1.0e-6F);
}

TEST_CASE("CPU backend reports its backend name") {
    const std::unique_ptr<vector_search::SearchBackend> backend =
        vector_search::create_backend("cpu");

    REQUIRE_EQ(backend->name(), std::string("cpu"));
}
