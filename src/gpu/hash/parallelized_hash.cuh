#ifndef GPU_CRACK_PARALLELIZED_HASH_CUH
#define GPU_CRACK_PARALLELIZED_HASH_CUH

#include <cstdio>
#include <ctime>

#include "commons.cuh"

Digest *parallelized_hash(Password *h_passwords, int passwordNumber);

Digest *parallelized_hash_time(Password *h_passwords, int passwordNumber, float *milliseconds);

#endif //GPU_CRACK_PARALLELIZED_HASH_CUH
