# The CMake setting of 'check/'

# CUDA
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler;-fopenmp)

# Set target
file(GLOB_RECURSE files "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
cuda_add_executable(nbfmm_test ${files} EXCLUDE_FROM_ALL)
include_directories(${CPPUNIT_INCLUDE})
target_link_libraries(nbfmm_test nbfmm_core ${GOMP_LIBRARY} ${CPPUNIT_LIBRARY})

# Add rule 'test'
add_custom_target(
  check
  COMMAND ./nbfmm_test
  DEPENDS nbfmm_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running test code"
)
