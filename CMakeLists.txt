cmake_minimum_required(VERSION 3.16)
project(noinfer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(glog REQUIRED)

set(link_lib glog pthread gtest)
set(link_math_lib armadillo blas lapack)
add_executable(noinfer main.cpp)

target_include_directories(noinfer PUBLIC /usr/include/armadillo_bits)
target_link_libraries(noinfer ${link_lib} ${link_math_lib})

enable_testing()
add_subdirectory(test)
