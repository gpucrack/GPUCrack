cmake_minimum_required(VERSION 3.19)
project(GPU_crack CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(src/gpu/hash/hash_functions)

add_executable(Chains
        src/gpu/chains.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/rainbow.cu
        src/gpu/hash/commons.cu)

add_executable(benchmarkTest
        src/gpu/hash/test/benchmarkTest.cu
        src/gpu/hash/hash_functions/cudaMd5.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/hash/commons.cu
        src/gpu/hash/commons.cuh)

add_executable(generalTest
        src/gpu/hash/commons.cu
        src/gpu/hash/commons.cuh
        src/gpu/hash/test/generalTest.cu
        src/gpu/hash/test/generalTest.cuh
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/hash/hash_functions/ntlm.cuh)

add_executable(complianceTest
        src/gpu/hash/commons.cu
        src/gpu/hash/commons.cuh
        src/gpu/hash/hash_functions/classicMd5.cu
        src/gpu/hash/hash_functions/classicMd5.cuh
        src/gpu/hash/test/complianceTest.cu
        src/gpu/hash/test/complianceTest.cuh
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/hash/hash_functions/ntlm.cuh)


set_target_properties(Chains PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(benchmarkTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(generalTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(complianceTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

