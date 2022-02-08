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
__device__ void reduce_digest2(unsigned long index, Digest * digest, Password  *plain_text);

/*
 * Reduces every digest of an array into plain texts on GPU.
 * Every thread of the GPU will compute a single reduction.
 * digests: the digest array to reduce
 * plain_texts: the generated reductions
 */
__global__ void reduce_digests2(Digest *digests, Password *plain_texts);

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
void display_reductions(Digest **digests, Password **passwords, int n = DEFAULT_PASSWORD_NUMBER);

__host__ void reduceKernel(int passwordNumber, int numberOfPass, int batchSize, float *milliseconds,
                           Password **h_passwords, Digest **h_results, int threadPerBlock);

__host__ void
reduce(Password *h_passwords, Digest *h_results, int passwordNumber, int numberOfPass, int threadsPerBlock);