#ifndef GPU_CRACK_CHAINS_CUH
#define GPU_CRACK_CHAINS_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "commons.cuh"
#include "rainbow.cuh"

__global__ void ntlm_chain_kernel2(Password * passwords, Digest * digests, int chainLength);
__device__ void reduce_digest(unsigned int index, Digest * digest, Password  * plain_text);
__host__ void generateChains(Password * h_passwords, Digest * h_results, int passwordNumber, int numberOfPass);
__host__ void chainKernel(int passwordNumber, int numberOfPass, int batchSize, float *milliseconds,
                          Password ** h_passwords, Digest ** h_results, int threadPerBlock,
                          int chainLength);

#endif //GPU_CRACK_CHAINS_CUH
