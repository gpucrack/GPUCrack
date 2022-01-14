#ifndef CUDA_NAIVE_EXHAUSTIVE_SEARCH_TEST_HASH_CUH
#define CUDA_NAIVE_EXHAUSTIVE_SEARCH_TEST_HASH_CUH

#include <cstdio>
#include <cstdlib>

#include "../../constants.cuh"
#include "cudaMd5.cuh"
#include "complianceTest.cuh"
#include <classicMd5.cuh>

#define REFERENCE_SENTENCE1 "The quick brown fox jumps over the lazy dog"
#define REFERENCE_RESULT1 "9e107d9d372bb6826bd81d3542a419d6"
#define NUMBER_OF_PASSWORD 1

int execute_tests();

#endif //CUDA_NAIVE_EXHAUSTIVE_SEARCH_TEST_HASH_CUH
