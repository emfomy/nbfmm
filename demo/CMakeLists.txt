# The CMake setting of 'demo/'

# Set target
cuda_add_executable(demo demo.cu)
target_link_libraries(demo nbfmm_core)

# Add rule 'run'
add_custom_target(
  run
  COMMAND ./demo
  DEPENDS demo
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running demo"
)
