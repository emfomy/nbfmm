# The CMake setting of 'doxygen/'

# Create configure file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile"
  "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
)

# Add rule
add_custom_target(
  dox
  COMMAND doxygen "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
)
