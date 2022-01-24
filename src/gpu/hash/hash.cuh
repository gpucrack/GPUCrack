#ifndef GPU_CRACK_HASH_CUH
#define GPU_CRACK_HASH_CUH

#include <cstdio>
#include <ctime>

#include "commons.cuh"

void hash(Password *h_passwords, Digest * h_results, int passwordNumber, int numberOfPass);

void hashTime(Password *h_passwords, Digest * h_results, int passwordNumber, float *milliseconds,
              int threadPerBlock, int numberOfPass);

#endif //GPU_CRACK_HASH_CUH
