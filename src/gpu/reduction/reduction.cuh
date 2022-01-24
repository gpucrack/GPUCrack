#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <iostream>

// The length of the hash function.
#define HASH_LENGTH 16

// The length of a password.
#define PASSWORD_LENGTH 7

#define CHARSET_LENGTH 62 // the number of characters in the charset

// How many thread per block to launch the kernel
// MUST BE A POWER OF 2
#define THREAD_PER_BLOCK 512

#define DEFAULT_PASSWORD_NUMBER 1000

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

// Reduces a hash into a plain text of length PLAIN_LENGTH.
void reduce(unsigned long int index, const char *hash, char *plain);
__global__ void reduce_kernel(int index, Digest *digests, Password *passwords);