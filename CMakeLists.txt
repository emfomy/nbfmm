# The main CMake setting

project(NBFMM)

cmake_minimum_required(VERSION 3.0)

# Set version number
set(NBFMM_VERSION_MAJOR 0)
set(NBFMM_VERSION_MINOR 1)
set(NBFMM_VERSION_PATCH 2)

# Set Makefile
set(CMAKE_COLOR_MAKEFILE ON)
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# CUDA
find_package(CUDA 7.5 REQUIRED)
set(FLAGS "-Wall,-Wextra")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-arch=sm_35;-std=c++11;-Xcompiler;${FLAGS})
include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")
link_directories("${PROJECT_BINARY_DIR}/library")

# Add subdirectories
add_subdirectory(include)
add_subdirectory(library)
add_subdirectory(demo)
add_subdirectory(doxygen)
