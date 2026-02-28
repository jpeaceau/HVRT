cmake_minimum_required(VERSION 3.20)
project(HVRT VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --------------------------------------------------------------------------
# Compiler flags
# --------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/O2 /W3)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
endif()

# --------------------------------------------------------------------------
# FetchContent dependencies
# --------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(Eigen pybind11)

# --------------------------------------------------------------------------
# hvrt_core — static library (no Python dependency)
# --------------------------------------------------------------------------
add_library(hvrt_core STATIC
    src/whitener.cpp
    src/target.cpp
    src/binner.cpp
    src/tree.cpp
    src/reduce.cpp
    src/expand.cpp
    src/hvrt.cpp
)

target_include_directories(hvrt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pcg
)

target_link_libraries(hvrt_core PUBLIC Eigen3::Eigen)

# --------------------------------------------------------------------------
# _hvrt — pybind11 extension module
# --------------------------------------------------------------------------
pybind11_add_module(_hvrt_cpp python/bindings.cpp)
target_link_libraries(_hvrt_cpp PRIVATE hvrt_core)
target_include_directories(_hvrt_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pcg
)

# Install into the hvrt package directory so it's importable as hvrt._hvrt_cpp
# SKBUILD_PROJECT_NAME is set by scikit-build-core; falls back gracefully otherwise.
if(DEFINED SKBUILD)
    install(TARGETS _hvrt_cpp DESTINATION hvrt)
endif()

# --------------------------------------------------------------------------
# Tests (optional — requires Catch2)
# --------------------------------------------------------------------------
option(HVRT_BUILD_TESTS "Build C++ unit tests" OFF)
if(HVRT_BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(hvrt_tests tests/test_correctness.cpp)
    target_link_libraries(hvrt_tests PRIVATE hvrt_core Catch2::Catch2WithMain)
    target_include_directories(hvrt_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pcg
    )
endif()
