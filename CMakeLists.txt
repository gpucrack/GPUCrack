cmake_minimum_required(VERSION 3.19)
project(GPU_crack CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(src/gpu/hash/hash_functions)

add_executable(Main_executable src/gpu/hash/parallelized_hash.cu src/gpu/hash/hash_functions/cudaMd5.cu src/gpu/hash/hash_functions/ntlm.cu src/gpu/hash/commons.cu src/gpu/hash/commons.cuh)
add_executable(Chains src/gpu/chains.cu src/gpu/hash/hash_functions/ntlm.cu src/gpu/rainbow.cu src/gpu/hash/commons.cu)
add_executable(Benchmark_executable src/gpu/hash/benchmark.cu src/gpu/hash/hash_functions/cudaMd5.cu src/gpu/hash/hash_functions/ntlm.cu src/gpu/hash/commons.cu src/gpu/hash/commons.cuh)
add_executable(Test_executable src/gpu/hash/test/complianceTest.cu src/gpu/hash/test/complianceTest.cuh src/gpu/hash/hash_functions/classicMd5.cu src/gpu/hash/hash_functions/classicMd5.cuh src/gpu/hash/commons.cu src/gpu/hash/commons.cuh src/gpu/hash/test/generalTest.cu src/gpu/hash/test/generalTest.cuh)

set_target_properties(Main_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Chains PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Benchmark_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Test_executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
