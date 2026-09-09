#include "dataset.hpp"
#include "test_framework.hpp"
#include "vector_search.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
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

TEST_CASE("cuda-naive stage timing preserves CPU-equivalent results") {
    const vector_search::Dataset dataset =
        vector_search::generate_synthetic_dataset({257, 7, 2, 2026});
    const vector_search::SearchRequest request{
        dataset.database.data(), dataset.queries.data(), 257, 7, 2, 5};

    const std::unique_ptr<vector_search::SearchBackend> cpu =
        vector_search::create_backend("cpu");
    const std::unique_ptr<vector_search::SearchBackend> cuda =
        vector_search::create_backend("cuda-naive");
    cuda->set_timing_breakdown_enabled(true);

    const vector_search::SearchResult expected = cpu->search(request);
    const vector_search::SearchResult actual = cuda->search(request);
    require_results_match(expected, actual, 2.0e-5F);

    const vector_search::BackendTiming timing = cuda->last_timing();
    REQUIRE(timing.has_kernel_latency);
    REQUIRE(timing.has_stage_timing);
    REQUIRE(std::isfinite(timing.allocation_milliseconds));
    REQUIRE(std::isfinite(timing.h2d_database_milliseconds));
    REQUIRE(std::isfinite(timing.h2d_queries_milliseconds));
    REQUIRE(std::isfinite(timing.kernel_latency_milliseconds));
    REQUIRE(std::isfinite(timing.d2h_scores_milliseconds));
    REQUIRE(std::isfinite(timing.cpu_topk_milliseconds));
    REQUIRE(std::isfinite(timing.cleanup_milliseconds));
    REQUIRE(std::isfinite(timing.total_e2e_milliseconds));
    REQUIRE(timing.total_e2e_milliseconds > 0.0);
}

TEST_CASE("cuda-block matches CPU on deterministic known vectors") {
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
        database.data(), queries.data(), 5, 3, 2, 3};

    const std::unique_ptr<vector_search::SearchBackend> cpu =
        vector_search::create_backend("cpu");
    const std::unique_ptr<vector_search::SearchBackend> block =
        vector_search::create_backend("cuda-block");

    const vector_search::SearchResult expected = cpu->search(request);
    const vector_search::SearchResult actual = block->search(request);
    require_results_match(expected, actual, 2.0e-6F);
    REQUIRE_EQ(actual.at(0, 0).index, 0U);
    REQUIRE_EQ(actual.at(1, 0).index, 2U);
}

TEST_CASE("cuda-warp matches CPU on deterministic known vectors") {
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
        database.data(), queries.data(), 5, 3, 2, 3};

    const std::unique_ptr<vector_search::SearchBackend> cpu =
        vector_search::create_backend("cpu");
    const std::unique_ptr<vector_search::SearchBackend> warp =
        vector_search::create_backend("cuda-warp");

    const vector_search::SearchResult expected = cpu->search(request);
    const vector_search::SearchResult actual = warp->search(request);
    require_results_match(expected, actual, 2.0e-6F);
    REQUIRE_EQ(actual.at(0, 0).index, 0U);
    REQUIRE_EQ(actual.at(1, 0).index, 2U);
}

TEST_CASE("cuda-naive, cuda-block, and cuda-warp match CPU across dimension boundaries") {
    const std::size_t dimensions[] = {
        3, 31, 32, 33, 127, 128, 255, 256, 257, 768};
    const char* backend_names[] = {
        "cuda-naive", "cuda-block", "cuda-warp"};

    for (const std::size_t dimension : dimensions) {
        const vector_search::Dataset dataset =
            vector_search::generate_synthetic_dataset(
                {11, dimension, 2, 9000 + dimension});
        const vector_search::SearchRequest request{
            dataset.database.data(),
            dataset.queries.data(),
            dataset.num_vectors,
            dataset.dimension,
            dataset.num_queries,
            3};
        const std::unique_ptr<vector_search::SearchBackend> cpu =
            vector_search::create_backend("cpu");
        const vector_search::SearchResult expected = cpu->search(request);

        for (const char* backend_name : backend_names) {
            const std::unique_ptr<vector_search::SearchBackend> cuda =
                vector_search::create_backend(backend_name);
            const vector_search::SearchResult actual = cuda->search(request);
            require_results_match(expected, actual, 5.0e-5F);
        }
    }
}

TEST_CASE("cuda-warp supports a single query, K=1, and stage timing") {
    const vector_search::Dataset dataset =
        vector_search::generate_synthetic_dataset({257, 7, 1, 2026});
    const vector_search::SearchRequest request{
        dataset.database.data(), dataset.queries.data(), 257, 7, 1, 1};

    const std::unique_ptr<vector_search::SearchBackend> cpu =
        vector_search::create_backend("cpu");
    const std::unique_ptr<vector_search::SearchBackend> warp =
        vector_search::create_backend("cuda-warp");
    warp->set_timing_breakdown_enabled(true);

    const vector_search::SearchResult expected = cpu->search(request);
    const vector_search::SearchResult actual = warp->search(request);
    require_results_match(expected, actual, 2.0e-5F);

    const vector_search::BackendTiming timing = warp->last_timing();
    REQUIRE(timing.has_kernel_latency);
    REQUIRE(timing.has_stage_timing);
    REQUIRE(std::isfinite(timing.allocation_milliseconds));
    REQUIRE(std::isfinite(timing.h2d_database_milliseconds));
    REQUIRE(std::isfinite(timing.h2d_queries_milliseconds));
    REQUIRE(std::isfinite(timing.kernel_latency_milliseconds));
    REQUIRE(std::isfinite(timing.d2h_scores_milliseconds));
    REQUIRE(std::isfinite(timing.cpu_topk_milliseconds));
    REQUIRE(std::isfinite(timing.cleanup_milliseconds));
    REQUIRE(std::isfinite(timing.total_e2e_milliseconds));
    REQUIRE(timing.total_e2e_milliseconds > 0.0);
}

TEST_CASE("cuda-block supports K=1 and stage timing") {
    const vector_search::Dataset dataset =
        vector_search::generate_synthetic_dataset({257, 7, 2, 2026});
    const vector_search::SearchRequest request{
        dataset.database.data(), dataset.queries.data(), 257, 7, 2, 1};

    const std::unique_ptr<vector_search::SearchBackend> cpu =
        vector_search::create_backend("cpu");
    const std::unique_ptr<vector_search::SearchBackend> block =
        vector_search::create_backend("cuda-block");
    block->set_timing_breakdown_enabled(true);

    const vector_search::SearchResult expected = cpu->search(request);
    const vector_search::SearchResult actual = block->search(request);
    require_results_match(expected, actual, 2.0e-5F);

    const vector_search::BackendTiming timing = block->last_timing();
    REQUIRE(timing.has_kernel_latency);
    REQUIRE(timing.has_stage_timing);
    REQUIRE(std::isfinite(timing.allocation_milliseconds));
    REQUIRE(std::isfinite(timing.h2d_database_milliseconds));
    REQUIRE(std::isfinite(timing.h2d_queries_milliseconds));
    REQUIRE(std::isfinite(timing.kernel_latency_milliseconds));
    REQUIRE(std::isfinite(timing.d2h_scores_milliseconds));
    REQUIRE(std::isfinite(timing.cpu_topk_milliseconds));
    REQUIRE(std::isfinite(timing.cleanup_milliseconds));
    REQUIRE(std::isfinite(timing.total_e2e_milliseconds));
    REQUIRE(timing.total_e2e_milliseconds > 0.0);
}

TEST_CASE("cuda-warp-resident reuses one database across sequential query batches") {
    const vector_search::Dataset dataset =
        vector_search::generate_synthetic_dataset({37, 128, 3, 2026});
    const vector_search::SearchRequest first_request{
        dataset.database.data(), dataset.queries.data(), 37, 128, 1, 1};
    const vector_search::SearchRequest second_request{
        nullptr, dataset.queries.data() + 128, 37, 128, 2, 5};
    const vector_search::SearchRequest second_expected_request{
        dataset.database.data(), dataset.queries.data() + 128, 37, 128, 2, 5};

    const std::unique_ptr<vector_search::SearchBackend> cpu =
        vector_search::create_backend("cpu");
    const std::unique_ptr<vector_search::SearchBackend> resident_base =
        vector_search::create_backend("cuda-warp-resident");
    vector_search::ResidentSearchBackend* resident =
        dynamic_cast<vector_search::ResidentSearchBackend*>(
            resident_base.get());
    REQUIRE(resident != nullptr);
    resident->set_timing_breakdown_enabled(true);

    bool search_before_prepare_threw = false;
    try {
        (void)resident->search(first_request);
    } catch (const std::logic_error&) {
        search_before_prepare_threw = true;
    }
    REQUIRE(search_before_prepare_threw);

    resident->prepare_database({
        dataset.database.data(), dataset.num_vectors, dataset.dimension});
    const vector_search::ResidentDatabaseInfo info =
        resident->database_info();
    REQUIRE(info.is_prepared);
    REQUIRE_EQ(info.num_vectors, dataset.num_vectors);
    REQUIRE_EQ(info.dimension, dataset.dimension);
    REQUIRE_EQ(info.bytes, dataset.database.size() * sizeof(float));
    const vector_search::BackendTiming preparation_timing =
        resident->last_timing();
    REQUIRE(preparation_timing.has_database_preparation_timing);
    REQUIRE(std::isfinite(preparation_timing.database_allocation_milliseconds));
    REQUIRE(std::isfinite(preparation_timing.database_h2d_milliseconds));
    REQUIRE(std::isfinite(preparation_timing.prepare_total_milliseconds));

    const vector_search::SearchResult first_expected =
        cpu->search(first_request);
    const vector_search::SearchResult first_actual =
        resident->search(first_request);
    require_results_match(first_expected, first_actual, 2.0e-5F);
    const vector_search::BackendTiming first_query_timing =
        resident->last_timing();
    REQUIRE(first_query_timing.has_kernel_latency);
    REQUIRE(first_query_timing.has_query_stage_timing);
    REQUIRE(!first_query_timing.has_database_preparation_timing);

    const vector_search::SearchResult second_expected =
        cpu->search(second_expected_request);
    const vector_search::SearchResult second_actual =
        resident->search(second_request);
    require_results_match(second_expected, second_actual, 2.0e-5F);
    const vector_search::BackendTiming second_query_timing =
        resident->last_timing();
    REQUIRE(second_query_timing.has_query_stage_timing);
    REQUIRE(!second_query_timing.has_database_preparation_timing);

    resident->clear_database();
    REQUIRE(!resident->database_is_prepared());
    bool search_after_clear_threw = false;
    try {
        (void)resident->search(first_request);
    } catch (const std::logic_error&) {
        search_after_clear_threw = true;
    }
    REQUIRE(search_after_clear_threw);
}

TEST_CASE("cuda-warp-resident reloads database contents and rejects incompatible requests") {
    const vector_search::Dataset first_dataset =
        vector_search::generate_synthetic_dataset({23, 256, 1, 11});
    const vector_search::Dataset second_dataset =
        vector_search::generate_synthetic_dataset({23, 256, 1, 12});
    const vector_search::SearchRequest first_request{
        first_dataset.database.data(),
        first_dataset.queries.data(),
        23,
        256,
        1,
        3};
    const vector_search::SearchRequest second_request{
        second_dataset.database.data(),
        first_dataset.queries.data(),
        23,
        256,
        1,
        3};

    const std::unique_ptr<vector_search::SearchBackend> cpu =
        vector_search::create_backend("cpu");
    const std::unique_ptr<vector_search::SearchBackend> resident_base =
        vector_search::create_backend("cuda-warp-resident");
    vector_search::ResidentSearchBackend* resident =
        dynamic_cast<vector_search::ResidentSearchBackend*>(
            resident_base.get());
    REQUIRE(resident != nullptr);

    resident->prepare_database({
        first_dataset.database.data(), 23, 256});
    const vector_search::SearchResult first_expected = cpu->search(first_request);
    const vector_search::SearchResult first_actual =
        resident->search(first_request);
    require_results_match(first_expected, first_actual, 5.0e-5F);

    resident->reload_database({
        second_dataset.database.data(), 23, 256});
    const vector_search::SearchResult second_expected =
        cpu->search(second_request);
    const vector_search::SearchResult second_actual =
        resident->search(second_request);
    require_results_match(second_expected, second_actual, 5.0e-5F);

    const vector_search::SearchRequest incompatible_request{
        second_dataset.database.data(),
        first_dataset.queries.data(),
        23,
        128,
        1,
        3};
    bool incompatible_request_threw = false;
    try {
        (void)resident->search(incompatible_request);
    } catch (const std::invalid_argument&) {
        incompatible_request_threw = true;
    }
    REQUIRE(incompatible_request_threw);

    resident->clear_database();
    REQUIRE(!resident->database_is_prepared());
}

TEST_CASE("cuda-warp-resident matches CPU at representative dimensions") {
    const std::size_t dimensions[] = {128, 256, 768};
    for (const std::size_t dimension : dimensions) {
        const vector_search::Dataset dataset =
            vector_search::generate_synthetic_dataset(
                {11, dimension, 2, 5000 + dimension});
        const vector_search::SearchRequest request{
            dataset.database.data(),
            dataset.queries.data(),
            dataset.num_vectors,
            dataset.dimension,
            dataset.num_queries,
            3};

        const std::unique_ptr<vector_search::SearchBackend> cpu =
            vector_search::create_backend("cpu");
        const std::unique_ptr<vector_search::SearchBackend> resident_base =
            vector_search::create_backend("cuda-warp-resident");
        vector_search::ResidentSearchBackend* resident =
            dynamic_cast<vector_search::ResidentSearchBackend*>(
                resident_base.get());
        REQUIRE(resident != nullptr);
        resident->prepare_database({
            dataset.database.data(), dataset.num_vectors, dataset.dimension});
        const vector_search::SearchResult expected = cpu->search(request);
        const vector_search::SearchResult actual = resident->search(request);
        require_results_match(expected, actual, 5.0e-5F);
        resident->clear_database();
    }
}

int main() {
    return vector_search_test::run_all();
}
