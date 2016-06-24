# The CMake setting of 'main/'

# CUDA
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler;-fopenmp)

# Set target
cuda_add_executable(nbfmm_main main.cu)
target_link_libraries(nbfmm_main -Wl,--start-group ${NBFMM_LIBRARY} -Wl,--end-group ${GOMP_LIBRARY})

# Add rule 'run'
file(GLOB scripts ${CMAKE_CURRENT_SOURCE_DIR}/script/*.py)
foreach(script ${scripts})
  get_filename_component(scriptname ${script} NAME_WE)
  add_custom_target(
    run-${scriptname}
    COMMAND python ${script} ${CMAKE_CURRENT_BINARY_DIR}/nbfmm_main nbfmm.y4m
    DEPENDS nbfmm_main ${NBFMM_LIBRARY} ${script} run-directory
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/run"
    COMMENT "Running model ${scriptname}"
  )
endforeach()
add_custom_target(
  run
  DEPENDS run-rectangle
)
add_custom_target(
  run-directory
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/run"
)

# Add to clean rule
set_directory_properties(
  PROPERTIES
  ADDITIONAL_MAKE_CLEAN_FILES
  "${CMAKE_BINARY_DIR}/run"
)

# Set target
cuda_add_executable(nbfmm_demo demo.cu)
target_link_libraries(nbfmm_demo -Wl,--start-group ${NBFMM_LIBRARY} ${GOMP_LIBRARY})

# Add rule 'demo'
add_custom_target(
  demo
  COMMAND ./nbfmm_demo
  DEPENDS nbfmm_demo ${NBFMM_LIBRARY}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running demo"
)
