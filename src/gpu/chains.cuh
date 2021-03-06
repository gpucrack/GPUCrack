#ifndef GPU_CRACK_CHAINS_CUH
#define GPU_CRACK_CHAINS_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <c++/9/fstream>

#include "commons.cuh"

/**
 * Main kernel used by GPU to hash and reduce
 * @param passwords : Password array used by both reduction and hash
 * @param digests : Digest array used by both reduction and hash
 * @param chainLength : chainLength = how many columns in the chain
 * @param pwd_length the length of a password (in characters).
 */
__global__ void
ntlmChainKernel(Password *passwords, Digest *digests, int chainLength, int pwd_length, unsigned long long domain);

__global__ void
ntlmChainKernelDebug(Password *passwords, Digest *digests, int chainLength, int pwd_length, unsigned long long domain);

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
 * @param pwd_length the length of a password (in characters).
 * @param start_path the path to the start point file that will be created.
 * @param end_path the path to the end point file that will be created.
 */
__host__ void
generateChains(Password *h_passwords, unsigned long long passwordNumber, int numberOfPass, int numberOfColumn,
               bool save, int theadsPerBlock, bool debug, bool debugKernel, Digest *h_results, int pwd_length,
               char *start_path, char *end_path, float *totalGPU, int batchNumber);

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
 * @param pwd_length the length of a password (in characters).
 * @param start_path the path to the start point file that will be created.
 * @param end_path the path to the end point file that will be created.
 */
__host__ void
chainKernel(unsigned long long passwordNumber, int numberOfPass, unsigned long long batchSize, float *milliseconds,
            Password **h_passwords, int threadPerBlock, int chainLength, bool debug, Digest **h_results,
            int pwd_length, char *start_path, char *end_path, bool kernelDebug);

#endif //GPU_CRACK_CHAINS_CUH
