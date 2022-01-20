#ifndef CUDA_NAIVE_EXHAUSTIVE_SEARCH_COMMONS_CUH
#define CUDA_NAIVE_EXHAUSTIVE_SEARCH_COMMONS_CUH

#include "../constants.cuh"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "./hash_functions/ntlm.cuh"

__host__ double memoryAnalysis(int passwordNumber);
__host__ int computeBatchSize(double numberOfPass, int passwordNumber);
__host__ void kernel(double numberOfPass, int batchSize, float * milliseconds, const clock_t * program_start,
                     Digest ** h_results, Password **h_passwords, int passwordNumber);
__host__ Password * generatePasswords(int passwordNumber);


#endif //CUDA_NAIVE_EXHAUSTIVE_SEARCH_COMMONS_CUH
