# The CMake setting of 'library/'

file(
  GLOB directory "${CMAKE_SOURCE_DIR}/source/nbfmm/*"
)
foreach(dir IN LISTS directory)
  if(IS_DIRECTORY ${dir})
    get_filename_component(dirname ${dir} NAME)
    file(GLOB_RECURSE files "${dir}/*.cu")
    cuda_add_library("nbfmm_${dirname}" ${files})
  endif()
endforeach()
