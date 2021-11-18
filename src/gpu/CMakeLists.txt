cmake_minimum_required(VERSION 3.19)
project(CUDA_naive_exhaustive_search CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDA_naive_exhaustive_search parallelized_hash.cu)
add_executable(ParaHash parahash.cu)

set_target_properties(CUDA_naive_exhaustive_search PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(ParaHash PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
