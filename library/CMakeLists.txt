# The CMake setting of 'library/'

macro(subdirlist result dir)
  file(GLOB children RELATIVE ${dir} ${dir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${dir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

set(libraries "")
set(sourcedir "${CMAKE_SOURCE_DIR}/source")
subdirlist(dirs "${sourcedir}")
foreach(dir IN LISTS dirs)
  subdirlist(subdirs "${sourcedir}/${dir}")
  foreach(subdir IN LISTS subdirs)
    file(GLOB_RECURSE files "${sourcedir}/${dir}/${subdir}/*.cu")
    cuda_add_library("${dir}_${subdir}" ${files})
    list(APPEND libraries "${dir}_${subdir}")
  endforeach()
endforeach()

# list(REMOVE_ITEM libraries "nbfmm_core")
# list(APPEND libraries "nbfmm_core")

set(NBFMM_LIBRARY ${libraries} PARENT_SCOPE)
