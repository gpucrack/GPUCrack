#ifndef GPU_CRACK_COMMONS_CUH
#define GPU_CRACK_COMMONS_CUH

#include "constants.cuh"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "./hash/hash_functions/ntlm.cuh"
#include "./chainsV2.cuh"
#include "./hash/hash.cuh"

// Generates passwordNumber random passwords, using a 62 character alphanumeric charset.
// The charset contains [a-zA-Z0-9].
__host__ void generateNewPasswords(Password ** result, int passwordNumber);

__host__ void generatePasswords(Password ** result, int passwordNumber);

// Returns the number of batches that we need to do
__host__ int memoryAnalysis(int passwordNumber);

// Returns the size a batch should have
__host__ int computeBatchSize(int numberOfPass, int passwordNumber);

__host__ void initEmptyArrays(Password ** passwords, Digest ** results, int passwordNumber);

__host__ void initArrays(Password ** passwords, Digest ** results, int passwordNumber);

// Launches the ntlm_kernel function (from ./hash_functions/ntlm.cuh), which hashes the specified number of passwords using NTLM.
__host__ void hashKernel(const int numberOfPass, int batchSize, float *milliseconds, const clock_t *program_start,
                         Digest **h_results, Password **h_passwords, int passwordNumber, int threadPerBlock);

__device__ __host__ void printDigest(Digest * dig);

__device__ __host__ void printPassword(Password * pwd);
#endif //GPU_CRACK_COMMONS_CUH
