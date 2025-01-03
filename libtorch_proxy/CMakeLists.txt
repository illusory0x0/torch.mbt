cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

# export compiler commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create shared library instead of an executable
add_library(tchproxy SHARED libproxy.cpp)

# Link with Torch libraries
target_link_libraries(tchproxy "${TORCH_LIBRARIES}")

# Set C++ standard
set_property(TARGET tchproxy PROPERTY CXX_STANDARD 17)