cmake_minimum_required(VERSION 3.20)
project(PhilemonTSH
    VERSION 0.1.0
    DESCRIPTION "Tiered Heterogeneous Memory Temporal Subgraph Indexing System"
    LANGUAGES CXX
)

# ─── C++ Standard ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Compiler flags ──────────────────────────────────────────────────
add_compile_options(
    -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
    -pthread
)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DPHILE_DEBUG=1")

# ─── Philemon header-only library ────────────────────────────────────
# All src/ headers are header-only; we expose them as an INTERFACE target
add_library(philemon_headers INTERFACE)
target_include_directories(philemon_headers INTERFACE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/bridge
    ${CMAKE_SOURCE_DIR}/src/index
    ${CMAKE_SOURCE_DIR}/src/wrapper
    ${CMAKE_SOURCE_DIR}/src/algorithms
    ${CMAKE_SOURCE_DIR}/src/executor
    ${CMAKE_SOURCE_DIR}/src/debug
    ${CMAKE_SOURCE_DIR}/src/loader
    ${CMAKE_SOURCE_DIR}/src/cost_model
    ${CMAKE_SOURCE_DIR}/src/prefetch
    ${CMAKE_SOURCE_DIR}/src/eviction
    ${CMAKE_SOURCE_DIR}/src/rebalance
    ${CMAKE_SOURCE_DIR}/src/compaction
    ${CMAKE_SOURCE_DIR}/src/scheduler
    ${CMAKE_SOURCE_DIR}/src/preprocessor
    ${CMAKE_SOURCE_DIR}/src/readers
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/types
    ${CMAKE_SOURCE_DIR}/src/entry
    ${CMAKE_SOURCE_DIR}/src/driver
)
target_link_libraries(philemon_headers INTERFACE pthread)

# ─── Benchmarks ──────────────────────────────────────────────────────
add_executable(philemon_bench      src/bench/philemon_bench.cpp)
add_executable(integration_bench   src/bench/integration_bench.cpp)
add_executable(ldbc_bench          src/bench/ldbc_bench.cpp)
add_executable(cross_tier_bench    src/bench/cross_tier_bench.cpp)
add_executable(phase4_engine_bench src/bench/phase4_engine_bench.cpp)

target_link_libraries(philemon_bench      PRIVATE philemon_headers)
target_link_libraries(integration_bench   PRIVATE philemon_headers)
target_link_libraries(ldbc_bench          PRIVATE philemon_headers)
target_link_libraries(cross_tier_bench    PRIVATE philemon_headers)
target_link_libraries(phase4_engine_bench PRIVATE philemon_headers)

# ─── Testing ─────────────────────────────────────────────────────────
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    include(GoogleTest)

    # Helper: add a Philemon test
    function(philemon_add_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE
            philemon_headers
            GTest::gtest_main
        )
        gtest_discover_tests(${TEST_NAME})
    endfunction()

    # ── M036: Core module tests ──
    philemon_add_test(test_core        test/test_core.cpp)

    # ── M037: Index module tests ──
    philemon_add_test(test_index       test/test_index.cpp)

    # ── M038: Algorithm tests ──
    philemon_add_test(test_algorithms  test/test_algorithms.cpp)

    # ── M039: Wrapper / reader / config tests ──
    philemon_add_test(test_wrapper     test/test_wrapper.cpp)

    # ── M040: Integration test ──
    philemon_add_test(test_integration test/test_integration.cpp)
endif()
