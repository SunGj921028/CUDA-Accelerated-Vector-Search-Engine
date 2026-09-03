cmake_minimum_required(VERSION 3.18)

project(CUDAAcceleratedVectorSearch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VECTOR_SEARCH_BUILD_TESTS "Build the CPU correctness tests" ON)

add_library(vector_search_core
    src/common/benchmark.cpp
    src/common/dataset.cpp
    src/cpu/cpu_search.cpp
    src/cpu/cpu_topk.cpp
)

target_include_directories(vector_search_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(vector_search_core PRIVATE $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(vector_search_core PRIVATE $<$<CONFIG:Release>:-O3>)
endif()

add_executable(vector_search src/main.cpp)
target_link_libraries(vector_search PRIVATE vector_search_core)

if(MSVC)
    target_compile_options(vector_search PRIVATE $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(vector_search PRIVATE $<$<CONFIG:Release>:-O3>)
endif()

if(VECTOR_SEARCH_BUILD_TESTS)
    enable_testing()

    add_executable(vector_search_tests
        tests/test_cpu.cpp
        tests/test_correctness.cpp
    )
    target_link_libraries(vector_search_tests PRIVATE vector_search_core)

    if(MSVC)
        target_compile_options(vector_search_tests PRIVATE $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(vector_search_tests PRIVATE $<$<CONFIG:Release>:-O3>)
    endif()

    add_test(NAME vector_search_cpu_tests COMMAND vector_search_tests)
endif()
