#ifndef GPU_CRACK_HASH_CUH
#define GPU_CRACK_HASH_CUH

#include <cstdio>
#include <ctime>

#include "../commons.cuh"

/**
 * Function called to hash
 * @param h_passwords : CPU Password array used to generate chains
 * @param h_results : CPU Digest array used to generate chains
 * @param passwordNumber : How many password we use as input (m0)
 * @param numberOfPass : How many passes we need to do to compute all batches
 * @param noPrint : Boolean to say if we print or not
 */
void hash(Password *h_passwords, Digest *h_results, int passwordNumber, int numberOfPass, bool noPrint);

/**
 * Same function as hash but we take milliseconds to measure GPU time and threadPerBlock for benchmark
 * @param h_passwords : CPU Password array used to generate chains
 * @param h_results : CPU Digest array used to generate chains
 * @param passwordNumber : How many password we use as input (m0)
 * @param milliseconds : Measure GPU time
 * @param numberOfPass : How many passes we need to do to compute all batches
 * @param threadPerBlock : How many threads per block we will use
 * @param numberOfPass : How many passes we need to do to compute all batches
 */
void hashTime(Password *h_passwords, Digest * h_results, int passwordNumber, float *milliseconds,
              int threadPerBlock, int numberOfPass);

/**
 * Function called by hash to call the hash kernel
 * @param numberOfPass : How many passes we need to do to compute all batches
 * @param batchSize : Size of the batch
 * @param milliseconds : Measure GPU time
 * @param program_start : Time at which the program started, used to display time
 * @param h_results : CPU Digest array used to generate chains
 * @param h_passwords : CPU Password array used to generate chains
 * @param passwordNumber : How many password we use as input (m0)
 * @param threadPerBlock : How many threads per block we will use
 */
__host__ void hashKernel(const int numberOfPass, int batchSize,
                         float *milliseconds, const clock_t *program_start,
                         Digest **h_results, Password **h_passwords, int passwordNumber,
                         int threadPerBlock);

#endif //GPU_CRACK_HASH_CUH
