#ifndef GPU_CRACK_CHAINSV2_CUH
#define GPU_CRACK_CHAINSV2_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "commons.cuh"
#include "rainbow.cuh"

__host__ int createChain();
__global__ void ntlm_chain_kernel2(int t);

#endif //GPU_CRACK_CHAINSV2_CUH
