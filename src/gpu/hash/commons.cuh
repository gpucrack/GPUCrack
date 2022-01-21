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

// Returns the number of batches that we need to do
__host__ int memoryAnalysis(int passwordNumber);

// Returns the size a batch should have
__host__ int computeBatchSize(int numberOfPass, int passwordNumber);

// Launches the ntlm_kernel function (from ./hash_functions/ntlm.cuh), which hashes the specified number of passwords using NTLM.
__host__ void kernel(int numberOfPass, int batchSize, float *milliseconds, const clock_t *program_start,
                     Digest **h_results, Password **h_passwords, int passwordNumber, int threadPerBlock);

// Generates passwordNumber random passwords, using a 62 character alphanumeric charset.
// The charset contains [a-zA-Z0-9].
__host__ Password *generatePasswords(int passwordNumber);


#endif //CUDA_NAIVE_EXHAUSTIVE_SEARCH_COMMONS_CUH
