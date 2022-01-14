cmake_minimum_required(VERSION 3.19)
project(GPU_crack CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(hash/hash_functions)

add_executable(Main_executable hash/parallelized_hash.cu hash/hash_functions/cudaMd5.cu hash/hash_functions/ntlm.cu hash/commons.cu hash/commons.cuh)
add_executable(Chains chains.cu hash/hash_functions/ntlm.cu rainbow.cu hash/commons.cu)
add_executable(Benchmark_executable benchmark.cu hash/hash_functions/cudaMd5.cu hash/hash_functions/ntlm.cu hash/commons.cu hash/commons.cuh)
add_executable(Test_executable hash/test_hash.cu hash/test_hash.cuh hash/hash_functions/classicMd5.cu hash/hash_functions/classicMd5.cuh hash/commons.cu hash/commons.cuh)

set_target_properties(Main_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Chains PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Benchmark_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Test_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
