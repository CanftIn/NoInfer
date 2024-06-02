cmake_minimum_required(VERSION 3.16)
project(noinfer)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

set(link_lib glog::glog GTest::gtest)
if(!WIN32)
    set(link_lib "${link_lib} pthread")
endif()
set(link_math_lib  ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

aux_source_directory(./test DIR_TEST)
aux_source_directory(./src DIR_SOURCE)
aux_source_directory(./src/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./src/layer/details DIR_DETAIL_LAYER)
aux_source_directory(./src/parser DIR_PARSER)

add_executable(noinfer main.cpp ${DIR_TEST} ${DIR_SOURCE} ${DIR_DETAIL_LAYER} ${DIR_ABSTRACT_LAYER} ${DIR_PARSER} )
target_link_libraries(noinfer ${link_lib} ${OpenCV_LIBS} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(noinfer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(noinfer PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(noinfer PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(noinfer PUBLIC ./include)

enable_testing()