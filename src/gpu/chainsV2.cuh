#ifndef GPU_CRACK_CHAINSV2_CUH
#define GPU_CRACK_CHAINSV2_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "commons.cuh"
#include "rainbow.cuh"

__global__ void ntlm_chain_kernel2(Password * passwords, Digest * digests, int chainLength);
__host__ void generateChains(Password * h_passwords, Digest * h_results, int passwordNumber, int numberOfPass);

#endif //GPU_CRACK_CHAINSV2_CUH
