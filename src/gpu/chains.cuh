#ifndef GPU_CRACK_CHAINS_CUH
#define GPU_CRACK_CHAINS_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <c++/9/fstream>

#include "commons.cuh"
#include "rainbow.cuh"

#define CHARSET_LENGTH 62

__global__ void ntlm_chain_kernel(Password *passwords, Digest *digests, int chainLength);

__device__ void reduce_digest(unsigned int index, Digest *digest, Password *plain_text);

__host__ void
generateChains(Password *h_passwords, Digest *h_results, int passwordNumber, int numberOfPass, int numberOfColumn,
               bool save, int theadsPerBlock, bool debug);

/**
 * Initializes a progress bar to be displayed in the console.
 */
__device__ void initLoadingBar()

/**
 * Update the progress bar.
 */
__device__ void incrementLoadingBar()


__host__ void
chainKernel(int passwordNumber, int numberOfPass, int batchSize, float *milliseconds, Password **h_passwords,
            Digest **h_results, int threadPerBlock, int chainLength, bool save, bool debug);

#endif //GPU_CRACK_CHAINS_CUH
