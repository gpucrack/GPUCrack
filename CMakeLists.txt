cmake_minimum_required(VERSION 3.19)
project(GPU_crack CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(src/gpu/hash/hash_functions)

add_executable(Chains
        src/gpu/chains.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/rainbow.cu
        src/gpu/commons.cu)

add_executable(benchmarkTest
        src/gpu/hash/test/benchmarkTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/commons.cu
        src/gpu/hash/hash.cuh
        )

add_executable(generalTest
        src/gpu/commons.cu
        src/gpu/hash/test/generalTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/hash/hash.cuh)

add_executable(complianceTest
        src/gpu/commons.cu
        src/gpu/hash/test/complianceTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/hash/hash.cuh)


set_target_properties(Chains PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(benchmarkTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(generalTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(complianceTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

