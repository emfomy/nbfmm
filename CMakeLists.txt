# The main CMake setting

# Set project
project(NBFMM)
cmake_minimum_required(VERSION 3.0)

# Set version number
set(NBFMM_VERSION_MAJOR 1)
set(NBFMM_VERSION_MINOR 2)
set(NBFMM_VERSION_PATCH 0)

# Set Makefile
set(CMAKE_COLOR_MAKEFILE ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

# CUDA
find_package(CUDA 7.5 REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2;-arch=sm_35;-std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler;-Wall,-Wextra)
foreach(dir IN LISTS CUDA_INCLUDE_DIRS)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-isystem;"${dir}")
endforeach()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-isystem;"${PROJECT_SOURCE_DIR}/external/include")
include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")
link_directories("${PROJECT_BINARY_DIR}/library" "${PROJECT_BINARY_DIR}/external/library")

# CPPUNIT root
set(CPPUNIT_ROOT "" CACHE PATH "The root path of CppUnit.")

# CPPUNIT include
message(STATUS "Looking for CppUnit headers")
find_path(CPPUNIT_INCLUDE NAMES cppunit PATHS ${CPPUNIT_ROOT})
if(CPPUNIT_INCLUDE STREQUAL CPPUNIT_INCLUDE-NOTFOUND)
  message(FATAL_ERROR "Could NOT find CppUnit headers")
endif()
message(STATUS "Looking for CppUnit headers - found")

# CPPUNIT library
message(STATUS "Looking for CppUnit libraries")
find_library(CPPUNIT_LIBRARY NAMES cppunit PATHS ${CPPUNIT_ROOT})
if(CPPUNIT_LIBRARY STREQUAL CPPUNIT_LIBRARY-NOTFOUND)
  message(FATAL_ERROR "Could NOT find CppUnit libraries")
endif()
message(STATUS "Looking for CppUnit libraries - found")

# Add subdirectories
add_subdirectory(include)
add_subdirectory(library)
add_subdirectory(main)
add_subdirectory(check)
add_subdirectory(doxygen)
