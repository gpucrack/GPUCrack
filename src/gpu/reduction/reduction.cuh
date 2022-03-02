#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>

#include "../commons.cuh"

// The length of the password charset
#define CHARSET_LENGTH 62

// The length of the digest charset
#define DIGEST_CHARSET_LENGTH 16

/*
 * Fills digests with n pseudo-randomly generated digests.
 * digests: the digest array to fill.
 * n: the number of digests to be generated.
 */
void generate_digests_random(Digest **digests, int n);

/*
 * Reduces a digest into a plain text using the column index.
 * index: column index
 * digest: the digest to reduce
 * plain_text: the generated reduction
 */
__host__ __device__ void reduceDigest(unsigned int index, Digest *digest, Password *plain_text);

/*
 * Reduces every digest of an array into plain texts on GPU.
 * Every thread of the GPU will compute a single reduction.
 * digests: the digest array to reduce
 * plain_texts: the generated reductions
 */
__global__ void reduceDigests(Digest *digests, Password *plain_texts, int column);

/*
 * Compares two passwords.
 * return true if they are equal, false otherwise.
 */
inline int pwdcmp(Password &p1, Password &p2);

/*
 * Finds the number of duplicates in a password array
 */
int count_duplicates(Password **passwords, bool debug, int passwordNumber);

/*
 * Displays a reduction as "DIGEST --> PASSWORD"
 * digests: the array of digests
 * passwords: the array of passwords
 * n (optional): only display the n first recutions
 */
void display_reductions(Digest **digests, Password **passwords, int n);

/**
 * Function called by reduce to use reduce kernel
 * @param passwordNumber : How many password we use as input (m0)
 * @param numberOfPass : How many passes we need to do to compute all batches
 * @param batchSize : Size of the batch
 * @param milliseconds : Used to measure GPU time
 * @param h_passwords : CPU Password array used to generate chains
 * @param h_results : CPU Digest array used to generate chains
 * @param threadPerBlock : How many threads per block we will use
 */
__host__ void reduceKernel(int passwordNumber, int numberOfPass, int batchSize, float *milliseconds,
                           Password **h_passwords, Digest **h_results, int threadPerBlock);

/**
 * Function called to reduce
 * @param h_passwords : CPU Password array used to generate chains
 * @param h_results : CPU Digest array used to generate chains
 * @param passwordNumber : How many password we use as input (m0)
 * @param numberOfPass : How many passes we need to do to compute all batches
 * @param threadsPerBlock : How many threads per block we will use
 */
__host__ void
reduce(Password *h_passwords, Digest *h_results, int passwordNumber, int numberOfPass, int threadsPerBlock);