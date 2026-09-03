#include "dataset.hpp"
#include "test_framework.hpp"
#include "vector_search.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

TEST_CASE("normalize_vectors normalizes every non-zero row") {
    std::vector<float> values{
        3.0F, 4.0F, 0.0F,
        1.0F, 2.0F, 2.0F,
        0.0F, 0.0F, 0.0F,
    };

    vector_search::normalize_vectors(values, 3, 3);

    REQUIRE_NEAR(values[0], 0.6F, 1.0e-6F);
    REQUIRE_NEAR(values[1], 0.8F, 1.0e-6F);
    REQUIRE_NEAR(values[3], 1.0F / 3.0F, 1.0e-6F);
    REQUIRE_NEAR(values[4], 2.0F / 3.0F, 1.0e-6F);
    REQUIRE_NEAR(values[5], 2.0F / 3.0F, 1.0e-6F);
    REQUIRE_EQ(values[6], 0.0F);
    REQUIRE_EQ(values[7], 0.0F);
    REQUIRE_EQ(values[8], 0.0F);

    for (std::size_t row = 0; row < 2; ++row) {
        float squared_norm = 0.0F;
        for (std::size_t column = 0; column < 3; ++column) {
            const float component = values[row * 3 + column];
            squared_norm += component * component;
        }
        REQUIRE_NEAR(squared_norm, 1.0F, 1.0e-6F);
    }
}

TEST_CASE("compute_similarity_scores calculates FP32 dot products") {
    const std::vector<float> database{
        1.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
        -1.0F, 1.0F, 0.0F,
    };
    const std::vector<float> query{0.5F, 0.5F, 0.0F};

    const std::vector<float> scores = vector_search::compute_similarity_scores(
        database.data(), query.data(), 3, 3);

    REQUIRE_EQ(scores.size(), 3U);
    REQUIRE_NEAR(scores[0], 0.5F, 1.0e-6F);
    REQUIRE_NEAR(scores[1], 0.5F, 1.0e-6F);
    REQUIRE_NEAR(scores[2], 0.0F, 1.0e-6F);
}

TEST_CASE("select_top_k sorts by score and breaks ties by index") {
    const std::vector<float> scores{0.25F, 0.9F, 0.9F, -0.2F};

    const std::vector<vector_search::SearchHit> top_hits =
        vector_search::select_top_k(scores, 2);

    REQUIRE_EQ(top_hits.size(), 2U);
    REQUIRE_EQ(top_hits[0].index, 1U);
    REQUIRE_EQ(top_hits[1].index, 2U);
    REQUIRE_NEAR(top_hits[0].score, 0.9F, 1.0e-6F);
    REQUIRE_NEAR(top_hits[1].score, 0.9F, 1.0e-6F);
}

TEST_CASE("generate_synthetic_dataset is reproducible for a seed") {
    const vector_search::DatasetConfig config{5, 4, 2, 12345};

    const vector_search::Dataset first =
        vector_search::generate_synthetic_dataset(config);
    const vector_search::Dataset second =
        vector_search::generate_synthetic_dataset(config);

    REQUIRE_EQ(first.database, second.database);
    REQUIRE_EQ(first.queries, second.queries);
    REQUIRE_EQ(first.num_vectors, config.num_vectors);
    REQUIRE_EQ(first.dimension, config.dimension);
    REQUIRE_EQ(first.num_queries, config.num_queries);

    const vector_search::Dataset different_seed =
        vector_search::generate_synthetic_dataset({5, 4, 2, 12346});
    REQUIRE(first.database != different_seed.database);
    REQUIRE(first.queries != different_seed.queries);

    for (std::size_t row = 0; row < first.num_vectors; ++row) {
        float squared_norm = 0.0F;
        for (std::size_t column = 0; column < first.dimension; ++column) {
            const float component = first.database[row * first.dimension + column];
            squared_norm += component * component;
        }
        REQUIRE_NEAR(squared_norm, 1.0F, 2.0e-6F);
    }
}

int main() {
    return vector_search_test::run_all();
}
