# The CMake setting of 'demo/'

# Set complier flags
set(FLAGS "-Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS "-std=c11 ${FLAGS}")
set(CMAKE_CXX_FLAGS "-std=c++11 ${FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "-O2 -Wno-unknown-pragmas -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wno-unknown-pragmas -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fsanitize=address")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")

# Set target
add_executable(demo demo.cpp)
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include")

# Add rule 'run'
add_custom_target(
  run
  COMMAND ./demo
  DEPENDS demo
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running demo"
)
