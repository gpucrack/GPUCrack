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

/**
 * Main kernel used by GPU to hash and reduce
 * @param passwords : Password array used by both reduction and hash
 * @param digests : Digest array used by both reduction and hash
 * @param chainLength : chainLength = how many columns in the chain
 */
__global__ void ntlmChainKernel(Password *passwords, Digest *digests, int chainLength, unsigned long domain);

__global__ void ntlmChainKernelDebug(Password *passwords, Digest *digests, int chainLength, unsigned long domain);

/**
 * Main function called to generate chains
 * @param h_passwords : CPU Password array used to generate chains
 * @param h_results : CPU Digest array used to generate chains
 * @param passwordNumber : How many password we use as input (m0)
 * @param numberOfPass : How many passes we need to do to compute all batches
 * @param numberOfColumn : How many column there is in a chain
 * @param save : Boolean used to say if we want to save start and end points in .txt files (long operation)
 * @param theadsPerBlock : How many threads per block we will use
 * @param debug : Show debug print
 */
__host__ void
generateChains(Password *h_passwords, Digest *h_results, int passwordNumber, int numberOfPass, int numberOfColumn,
               bool save, int theadsPerBlock, bool debug, bool debugKernel);

/**
 * Function used to call chain Kernel inside generateChains
 * @param passwordNumber : How many password we use as input (m0)
 * @param numberOfPass : How many passes we need to do to compute all batches
 * @param batchSize : Size of the batch
 * @param milliseconds : Used to measure GPU time
 * @param h_passwords : CPU Password array used to generate chains
 * @param h_results : CPU Digest array used to generate chains
 * @param threadPerBlock : How many threads per block we will use
 * @param chainLength : chainLength = how many columns in the chain
 * @param save : Boolean used to say if we want to save start and end points in .txt files (long operation)
 * @param debug : Show debug print
 */
__host__ void
chainKernel(int passwordNumber, int numberOfPass, int batchSize, float *milliseconds, Password **h_passwords,
            Digest **h_results, int threadPerBlock, int chainLength, bool save, bool debug);

#endif //GPU_CRACK_CHAINS_CUH
