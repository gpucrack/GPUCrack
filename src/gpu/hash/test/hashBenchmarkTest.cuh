#ifndef GPU_CRACK_HASHBENCHMARKTEST_CUH
#define GPU_CRACK_HASHBENCHMARKTEST_CUH

#include <cstdio>

#include "../hash.cu"

#define NUMBER_OF_TEST 10
#define MAX_THREAD_NUMBER 1024

void benchmark(int passwordNumber);

#endif //GPU_CRACK_HASHBENCHMARKTEST_CUH
