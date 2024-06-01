cmake_minimum_required(VERSION 3.16)
project(noinfer)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)

set(link_lib glog::glog GTest::gtest)
if(!WIN32)
    set(link_lib "${link_lib} pthread")
endif()
set(link_math_lib  ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
aux_source_directory(./test DIR_TEST_ARMA)

add_executable(noinfer main.cpp ${DIR_TEST_ARMA})
target_link_libraries(noinfer ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(noinfer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(noinfer PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(noinfer PUBLIC ${Armadillo_INCLUDE_DIR})
enable_testing()