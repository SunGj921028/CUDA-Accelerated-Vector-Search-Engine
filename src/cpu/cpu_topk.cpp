#include "vector_search.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vector_search {

std::vector<SearchHit> select_top_k(
    const std::vector<float>& scores,
    std::size_t k) {
    if (k == 0) {
        throw std::invalid_argument("topk must be greater than zero");
    }
    if (k > scores.size()) {
        throw std::invalid_argument("topk must not exceed the number of scores");
    }

    std::vector<SearchHit> ranked;
    ranked.reserve(scores.size());
    for (std::size_t index = 0; index < scores.size(); ++index) {
        if (!std::isfinite(scores[index])) {
            throw std::invalid_argument("similarity scores must be finite");
        }
        ranked.push_back({index, scores[index]});
    }

    const auto better = [](const SearchHit& left, const SearchHit& right) {
        if (left.score != right.score) {
            return left.score > right.score;
        }
        return left.index < right.index;
    };

    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(), better);
    ranked.resize(k);
    return ranked;
}

}  // namespace vector_search
