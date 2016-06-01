# The CMake setting of 'library/'

file(
  GLOB_RECURSE FILES "${CMAKE_SOURCE_DIR}/source/*.cu"
)
cuda_add_library(nbfmm ${FILES})
