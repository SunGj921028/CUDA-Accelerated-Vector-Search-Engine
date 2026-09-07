#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace vector_search::cuda_detail {

inline std::string format_cuda_error(
    cudaError_t status,
    const char* operation) {
    std::ostringstream message;
    message << operation << " failed with "
            << cudaGetErrorName(status) << ": "
            << cudaGetErrorString(status);
    return message.str();
}

inline void check_cuda(
    cudaError_t status,
    const char* operation,
    const char* file,
    int line) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << file << ':' << line << ": "
                << format_cuda_error(status, operation);
        throw std::runtime_error(message.str());
    }
}

}  // namespace vector_search::cuda_detail

#define VECTOR_SEARCH_CUDA_CHECK(call) \
    ::vector_search::cuda_detail::check_cuda( \
        (call), #call, __FILE__, __LINE__)
