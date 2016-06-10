# The CMake setting of 'demo/'

# CUDA
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler;-fopenmp)

# Set target
cuda_add_executable(demo demo.cu)
target_link_libraries(demo nbfmm_core gomp)

# Add rule 'run'
add_custom_target(
  run
  COMMAND ./demo
  DEPENDS demo
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running demo"
)
