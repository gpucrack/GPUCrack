#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <time.h>


// The length of the hash function.
#define HASH_LENGTH 16

// The length of a password.
#define PASSWORD_LENGTH 7

// The length of the password charset
#define CHARSET_LENGTH 64

// The length of the digest charset
#define DIGEST_CHARSET_LENGTH 16

// How many thread per block to launch the hashKernel
// MUST BE A POWER OF 2
#define THREAD_PER_BLOCK 512

// One billion 1000000000
// One hundred million 100000000
// Ten millions 10000000
// One thousand 1000
#define DEFAULT_PASSWORD_NUMBER 1000000000

// A macro to have a ceil-like function.
#define CEILING(x, y) (((x) + (y)-1) / (y))

// A password put into a union. This is easier to use with malloc and crypto
// functions.
typedef union {
    uint8_t bytes[PASSWORD_LENGTH];
    uint32_t i[CEILING(PASSWORD_LENGTH, 4)];
} Password;

// A digest put into an union.
typedef union {
    uint8_t bytes[HASH_LENGTH];
    uint32_t i[CEILING(HASH_LENGTH, 4)];
} Digest;

/*
 * Displays a single password properly, char by char.
 * pwd: the password to display.
 */
void display_password(Password &pwd, bool br = true);

/*
 * Displays a password array properly with chars.
 * passwords: the password array to display.
 */
void display_passwords(Password **passwords);

/*
 * Displays a single digest properly.
 * digest: the digest to display.
 */
void display_digest(Digest &digest, bool br = true);

/*
 * Displays a digest array properly with chars.
 * digests: the digest array to display.
 */
void display_digests(Digest **digests);

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
__device__ void reduce_digest(unsigned long index, Digest &digest, Password &plain_text);

/*
 * Reduces every digest of an array into plain texts on GPU.
 * Every thread of the GPU will compute a single reduction.
 * digests: the digest array to reduce
 * plain_texts: the generated reductions
 */
__global__ void reduce_digests(Digest **digests, Password **plain_texts);

/*
 * Compares two passwords.
 * return true if they are equal, false otherwise.
 */
inline int pwdcmp(Password &p1, Password &p2);

/*
 * Finds the number of duplicates in a password array
 */
int count_duplicates(Password **passwords, bool debug = false);

/*
 * Displays a reduction as "DIGEST --> PASSWORD"
 * digests: the array of digests
 * passwords: the array of passwords
 * n (optional): only display the n first recutions
 */
void display_reductions(Digest **digests, Password **passwords, int n = DEFAULT_PASSWORD_NUMBER);