# The CMake setting of 'main/'

# CUDA
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler;-fopenmp)

# Set target
cuda_add_executable(nbfmm_main main.cu)
target_link_libraries(nbfmm_main ${NBFMM_LIBRARY} ${GOMP_LIBRARY})

# Add rule 'run'
add_custom_target(
  run
  COMMAND ./nbfmm_main
  DEPENDS nbfmm_main ${NBFMM_LIBRARY}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running main"
)

# Set target
cuda_add_executable(nbfmm_demo demo.cu)
target_link_libraries(nbfmm_demo ${NBFMM_LIBRARY} ${GOMP_LIBRARY})

# Add rule 'demo'
add_custom_target(
  demo
  COMMAND ./nbfmm_demo
  DEPENDS nbfmm_demo ${NBFMM_LIBRARY}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running demo"
)
