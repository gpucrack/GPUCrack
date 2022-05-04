cmake_minimum_required(VERSION 3.19)
project(GPU_crack CUDA CXX C)

set(CMAKE_CUDA_STANDARD 14)

include_directories(src/gpu/hash/hash_functions)

add_executable(hashGeneralTest
        src/gpu/commons.cu
        src/gpu/hash/test/hashGeneralTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/common/filtration.c
        src/gpu/chains.cu
        src/gpu/reduction/reduction.cu)

add_executable(reduction
        src/gpu/commons.cu
        src/gpu/reduction/reduction.cu
        src/gpu/reduction/test/reductionTest.cu
        src/gpu/reduction/test/reductionTest.cuh)

add_executable(chainTest
        src/gpu/chains.cu
        src/gpu/commons.cu
        src/gpu/test/chainTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/hash/hash.cu
        src/common/filtration.c
        src/gpu/chains.cu
        src/gpu/reduction/reduction.cu)

add_executable(chainCoverageTest
        src/gpu/chains.cu
        src/gpu/commons.cu
        src/gpu/test/chainCoverageTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/reduction/reduction.cu
        src/gpu/hash/hash.cu
        src/common/filtration.c)

add_executable(generateTableTest
        src/gpu/chains.cu
        src/gpu/commons.cu
        src/gpu/test/generateTableTest.cu
        src/gpu/hash/hash_functions/ntlm.cu
        src/gpu/reduction/reduction.cu
        src/common/filtration.c
        src/common/filtration2.cu)

add_executable(online
        src/common/online.c)

add_executable(tableCoverageTest
        src/common/test/tableCoverageTest.c
)

add_executable(filtrationTest
        src/common/test/filtrationTest.c
        src/common/filtration.c)

set_target_properties(hashGeneralTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(reduction PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(chainTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(chainCoverageTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(generateTableTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(tableCoverageTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(filtrationTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(online PRIVATE m)