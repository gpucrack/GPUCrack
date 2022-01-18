#ifndef GPU_CRACK_GENERALTEST_CUH
#define GPU_CRACK_GENERALTEST_CUH

#include "../../constants.cuh"
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "../parallelized_hash.cu"

__host__ Password * generatePasswords(long passwordNumber);

#endif //GPU_CRACK_GENERALTEST_CUH
