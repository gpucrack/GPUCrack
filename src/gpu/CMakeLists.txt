cmake_minimum_required(VERSION 3.19)
project(CUDA_naive_exhaustive_search CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Main_executable chains.cu myMd5.cu ntlm.cu rainbow.cu)
add_executable(Test_executable test_hash.cu test_hash.cuh md5.cu md5.cuh)

set_target_properties(Main_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Test_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
