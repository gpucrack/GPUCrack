cmake_minimum_required(VERSION 3.19)
project(CUDA_naive_exhaustive_search CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Main_executable parallelized_hash.cu hash_functions/cudaMd5.cu hash_functions/ntlm.cu)
add_executable(Benchmark_executable benchmark.cu hash_functions/cudaMd5.cu hash_functions/ntlm.cu)
add_executable(Test_executable test_hash.cu test_hash.cuh hash_functions/classicMd5.cu hash_functions/classicMd5.cuh)

set_target_properties(Main_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Benchmark_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Test_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
