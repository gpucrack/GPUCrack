#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <stdint.h>

// The length of the hash function.
#define HASH_LENGTH 16

// The length of a password.
#define PASSWORD_LENGTH 7

// The number of password to hash. (2^27)
#define PASSWORD_NUMBER 134217728

// How many thread per block to launch the kernel
#define THREAD_PER_BLOCK 128

// A macro to have a ceil-like function.
#define CEILING(x, y) (((x) + (y)-1) / (y))

// A password put into an union. This is easier to use with mallocs and crypto
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
#endif  // CONSTANTS_CUH
