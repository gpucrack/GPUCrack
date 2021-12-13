//
// Created by mynder on 13/12/2021.
//

#ifndef CUDA_NAIVE_EXHAUSTIVE_SEARCH_COMMONS_CUH
#define CUDA_NAIVE_EXHAUSTIVE_SEARCH_COMMONS_CUH


#include "constants.cuh"

class commons {

};

__host__ double memoryAnalysis();
__host__ int computeBatchSize(double numberOfPass);
__host__ void readPasswords(const Password * h_passwords, const int batchSize);
__host__ void kernel(const double numberOfPass, int batchSize, float * milliseconds, const clock_t * program_start,
                     Digest ** h_results);


#endif //CUDA_NAIVE_EXHAUSTIVE_SEARCH_COMMONS_CUH
