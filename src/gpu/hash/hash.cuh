#ifndef GPU_CRACK_HASH_CUH
#define GPU_CRACK_HASH_CUH

#include <cstdio>
#include <ctime>

#include "../commons.cuh"

void hash(Password *h_passwords, Digest *h_results, int passwordNumber, int numberOfPass, bool noPrint);

void hashTime(Password *h_passwords, Digest * h_results, int passwordNumber, float *milliseconds,
              int threadPerBlock, int numberOfPass);

__host__ void hashKernel(const int numberOfPass, int batchSize,
                         float *milliseconds, const clock_t *program_start,
                         Digest **h_results, Password **h_passwords, int passwordNumber,
                         int threadPerBlock);

#endif //GPU_CRACK_HASH_CUH
