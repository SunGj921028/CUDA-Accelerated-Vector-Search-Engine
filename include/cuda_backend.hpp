#pragma once

#include "vector_search.hpp"

#include <memory>
#include <string>

namespace vector_search {

struct CudaRuntimeInfo {
    bool available = false;
    std::string device_name;
    std::string toolkit_version;
    std::string error;
};

CudaRuntimeInfo query_cuda_runtime_info();

std::unique_ptr<SearchBackend> create_cuda_naive_backend();
std::unique_ptr<SearchBackend> create_cuda_block_backend();
std::unique_ptr<SearchBackend> create_cuda_warp_backend();
std::unique_ptr<ResidentSearchBackend> create_cuda_warp_resident_backend();

}  // namespace vector_search
