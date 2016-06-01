# The main CMake setting

project(LORASC)

cmake_minimum_required(VERSION 2.8)

# Set version number
set(NBFMM_VERSION_MAJOR 0)
set(NBFMM_VERSION_MINOR 1)
set(NBFMM_VERSION_PATCH dev)

# Set Makefile
set(CMAKE_COLOR_MAKEFILE ON)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# Set module paths
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Add subdirectories
add_subdirectory(demo)
add_subdirectory(doxygen)

# Create configure files
file(
  GLOB_RECURSE CONFIGFILES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.in"
)
foreach(CONFIGFILE IN LISTS CONFIGFILES)
  string(REGEX REPLACE "\\.[^.]*$" "" FILE ${CONFIGFILE})
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/${CONFIGFILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${FILE}"
  )
endforeach()
