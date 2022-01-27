#ifndef GPU_CRACK_CHAINSV2_CUH
#define GPU_CRACK_CHAINSV2_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "commons.cuh"
#include "rainbow.cuh"

__host__ int generateChains();
__global__ void ntlm_chain_kernel2(Password * d_passwords, Digest * d_results, int chainLength);

#endif //GPU_CRACK_CHAINSV2_CUH
