# The main CMake setting

project(LORASC)

cmake_minimum_required(VERSION 2.8)

# Set version number
set(NBFMM_VERSION_MAJOR 0)
set(NBFMM_VERSION_MINOR 1)
set(NBFMM_VERSION_PATCH dev)

# Set Makefile
set(CMAKE_COLOR_MAKEFILE ON)
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# CUDA
find_package(CUDA REQUIRED 7.5)
set(FLAGS "-Wall,-Wextra")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-arch=sm_35;-std=c++11;-Xcompiler;${FLAGS})
include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")
link_directories("${PROJECT_BINARY_DIR}/library")

# Add subdirectories
add_subdirectory(include)
add_subdirectory(library)
add_subdirectory(demo)
add_subdirectory(doxygen)
