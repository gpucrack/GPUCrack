#ifndef CUDA_NAIVE_EXHAUSTIVE_SEARCH_COMMONS_CUH
#define CUDA_NAIVE_EXHAUSTIVE_SEARCH_COMMONS_CUH

#include "../constants.cuh"

__host__ double memoryAnalysis();
__host__ int computeBatchSize(double numberOfPass);
__host__ void readPasswords(const Password * h_passwords, const int batchSize);
__host__ void kernel(double numberOfPass, int batchSize, float * milliseconds, const clock_t * program_start,
                     Digest ** h_results);


#endif //CUDA_NAIVE_EXHAUSTIVE_SEARCH_COMMONS_CUH