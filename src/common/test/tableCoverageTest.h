#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint-gcc.h>

// The password length in the rainbow tables.
#define PASSWORD_LENGTH 3

// The length of the charset.
#define CHARSET_LENGTH 62

// The length of the digest produced by the hash function (NTLM).
#define HASH_LENGTH 16

// A macro to have a ceil-like function.
#define CEILING(x, y) (((x) + (y)-1) / (y))


// A password put into a union. This is easier to use with malloc and crypto
// functions.
typedef struct {
    uint8_t bytes[PASSWORD_LENGTH];
} Password;

// A digest put into a union.
typedef union {
    uint8_t bytes[HASH_LENGTH];
    uint32_t i[CEILING(HASH_LENGTH, 4)];
    uint64_t value;
} Digest;