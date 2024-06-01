find_package(GTest REQUIRED)

aux_source_directory(../test DIR_TEST)
set(link_lib glog gtest pthread)
set(link_math_lib armadillo blas lapack)

add_executable(test_noinfer ${DIR_TEST})

target_link_libraries(test_noinfer ${link_lib} ${link_math_lib})
