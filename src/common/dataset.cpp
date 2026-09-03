#include "dataset.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace vector_search {
namespace {

std::size_t checked_product(std::size_t left, std::size_t right) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error("vector dimensions overflow size_t");
    }
    return left * right;
}

float next_synthetic_value(std::mt19937_64& generator) {
    const std::uint64_t random_value = generator();
    const long double unit = static_cast<long double>(random_value) /
                             static_cast<long double>(
                                 std::numeric_limits<std::uint64_t>::max());
    return static_cast<float>(unit * 2.0L - 1.0L);
}

}  // namespace

void normalize_vectors(
    float* data,
    std::size_t num_vectors,
    std::size_t dimension) {
    if (num_vectors == 0) {
        return;
    }
    if (dimension == 0) {
        throw std::invalid_argument("vector dimension must be greater than zero");
    }
    if (data == nullptr) {
        throw std::invalid_argument("vector data must not be null");
    }

    for (std::size_t row = 0; row < num_vectors; ++row) {
        float* vector = data + row * dimension;
        double squared_norm = 0.0;
        for (std::size_t column = 0; column < dimension; ++column) {
            if (!std::isfinite(vector[column])) {
                throw std::invalid_argument("vector data must be finite");
            }
            const double component = static_cast<double>(vector[column]);
            squared_norm += component * component;
        }

        if (squared_norm == 0.0) {
            continue;
        }

        const double norm = std::sqrt(squared_norm);
        for (std::size_t column = 0; column < dimension; ++column) {
            vector[column] = static_cast<float>(
                static_cast<double>(vector[column]) / norm);
        }
    }
}

void normalize_vectors(
    std::vector<float>& data,
    std::size_t num_vectors,
    std::size_t dimension) {
    if (data.size() != checked_product(num_vectors, dimension)) {
        throw std::invalid_argument(
            "vector data size does not match the requested shape");
    }
    normalize_vectors(data.data(), num_vectors, dimension);
}

Dataset generate_synthetic_dataset(const DatasetConfig& config) {
    if (config.num_vectors == 0) {
        throw std::invalid_argument("num_vectors must be greater than zero");
    }
    if (config.dimension == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }
    if (config.num_queries == 0) {
        throw std::invalid_argument("num_queries must be greater than zero");
    }

    const std::size_t vector_elements =
        checked_product(config.num_vectors, config.dimension);
    const std::size_t query_elements =
        checked_product(config.num_queries, config.dimension);

    Dataset dataset;
    dataset.num_vectors = config.num_vectors;
    dataset.dimension = config.dimension;
    dataset.num_queries = config.num_queries;
    dataset.database.resize(vector_elements);
    dataset.queries.resize(query_elements);

    std::mt19937_64 generator(config.seed);
    for (float& value : dataset.database) {
        value = next_synthetic_value(generator);
    }
    for (float& value : dataset.queries) {
        value = next_synthetic_value(generator);
    }

    normalize_vectors(dataset.database, dataset.num_vectors, dataset.dimension);
    normalize_vectors(dataset.queries, dataset.num_queries, dataset.dimension);
    return dataset;
}

}  // namespace vector_search
