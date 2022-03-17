cmake_minimum_required(VERSION 3.19)
project(GPU_crack CUDA CXX C)

set(CMAKE_CUDA_STANDARD 14)

include_directories(src/gpu/hash/hash_functions)

add_executable(hashBenchmarkTest
        src/gpu/hash/test/hashBenchmarkTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/commons.cu
        src/gpu/hash/hash.cuh
        )

add_executable(hashGeneralTest
        src/gpu/commons.cu
        src/gpu/hash/test/hashGeneralTest.cu
        src/gpu/hash/hash_functions/ntlm.cu)

add_executable(hashComplianceTest
        src/gpu/commons.cu
        src/gpu/hash/test/hashComplianceTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/hash/hash.cuh)

add_executable(reductionCPU
        src/common/reduction/reduction.cpp)

add_executable(reduction
        src/gpu/commons.cu
        src/gpu/reduction/reduction.cu src/gpu/reduction/test/reductionTest.cu src/gpu/reduction/test/reductionTest.cuh)

add_executable(chainTest
        src/gpu/chains.cu
        src/gpu/commons.cu
        src/gpu/test/chainTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/reduction/reduction.cu
        src/gpu/hash/hash.cu)

add_executable(chainTotalCoverageTest
        src/gpu/chains.cu
        src/gpu/commons.cu
        src/gpu/test/chainTotalCoverageTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/reduction/reduction.cu
        src/gpu/hash/hash.cu)

add_executable(chainCoverageTest
        src/gpu/chains.cu
        src/gpu/commons.cu
        src/gpu/test/chainCoverageTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/reduction/reduction.cu
        src/gpu/hash/hash.cu)

add_executable(chainBenchmarkTest
        src/gpu/chains.cu
        src/gpu/commons.cu
        src/gpu/test/chainBenchmarkTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/reduction/reduction.cu)

add_executable(chainBenchmarkThreadTest
        src/gpu/chains.cu
        src/gpu/commons.cu
        src/gpu/test/chainBenchmarkThreadTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/reduction/reduction.cu)

add_executable(generateTableTest
        src/gpu/chains.cu
        src/gpu/commons.cu
        src/gpu/test/generateTableTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/reduction/reduction.cu
        src/common/filtration.c)

add_executable(online
        src/common/online.c)

add_executable(tableCoverageTest
        src/common/test/tableCoverageTest.c
)

set_target_properties(hashBenchmarkTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(hashGeneralTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(hashComplianceTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(reductionCPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(reduction PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(chainTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(chainTotalCoverageTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(chainCoverageTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(chainBenchmarkTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(chainBenchmarkThreadTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(generateTableTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(online PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(tableCoverageTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)