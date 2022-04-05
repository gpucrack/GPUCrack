#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <cstdint>

#define CHARSET_LENGTH 62

__device__ static const unsigned char charset[CHARSET_LENGTH] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                                                                 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                                                                 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                                                                 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

// The length of the hash function.
#define HASH_LENGTH 16

// The length of a password.
#define PASSWORD_LENGTH 4

// How many thread per block to launch the hashKernel
// MUST BE A POWER OF 2
#define THREAD_PER_BLOCK 32

// This is the maximum number of thread that we can used on a GPU
#define MAX_THREAD_NUMBER 1024

// A password put into a struct.
typedef struct {
    uint8_t bytes[PASSWORD_LENGTH];
} Password;

// A digest put into an union.
typedef union {
    uint8_t bytes[HASH_LENGTH];
    uint32_t i[HASH_LENGTH/4];
} Digest;
#endif  // CONSTANTS_CUH
