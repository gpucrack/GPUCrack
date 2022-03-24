#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <cstdint>

// The length of the hash function.
#define HASH_LENGTH 16

// The length of a password.
#define PASSWORD_LENGTH 3

// How many thread per block to launch the hashKernel
// MUST BE A POWER OF 2
#define THREAD_PER_BLOCK 32

// Default number of passwords (m0) to use
#define DEFAULT_PASSWORD_NUMBER 268435456

// A macro to have a ceil-like function.
#define CEILING(x, y) (((x) + (y)-1) / (y))

// This is the maximum number of thread that we can used on a GPU
#define MAX_THREAD_NUMBER 1024

// A password put into a struct.
typedef struct {
    uint8_t bytes[PASSWORD_LENGTH];
} Password;

// A digest put into an union.
typedef union {
    uint8_t bytes[HASH_LENGTH];
    uint32_t i[CEILING(HASH_LENGTH, 4)];
    uint64_t value;
} Digest;
#endif  // CONSTANTS_CUH
