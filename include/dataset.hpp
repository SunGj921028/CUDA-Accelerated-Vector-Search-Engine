#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vector_search {

struct DatasetConfig {
    std::size_t num_vectors = 10000;
    std::size_t dimension = 128;
    std::size_t num_queries = 1;
    std::uint64_t seed = 42;
};

struct Dataset {
    std::size_t num_vectors = 0;
    std::size_t dimension = 0;
    std::size_t num_queries = 0;
    std::vector<float> database;
    std::vector<float> queries;
};

// Normalize each row in a row-major array to unit L2 norm. A zero row is
// left unchanged so this operation is safe for sparse or partially-filled
// inputs.
void normalize_vectors(
    float* data,
    std::size_t num_vectors,
    std::size_t dimension);

void normalize_vectors(
    std::vector<float>& data,
    std::size_t num_vectors,
    std::size_t dimension);

// Generate normalized, deterministic FP32 database and query vectors.
Dataset generate_synthetic_dataset(const DatasetConfig& config);

}  // namespace vector_search
