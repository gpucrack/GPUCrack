#ifndef RAINBOW_H
#define RAINBOW_H

#define _CRT_SECURE_NO_WARNINGS

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint-gcc.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

// The password length in the rainbow tables.
#define PASSWORD_LENGTH 5

#define TEST_COVERAGE 10

// The length of the charset.
#define CHARSET_LENGTH 62

// The length of the digest produced by the hash function (NTLM).
#define HASH_LENGTH 16

// A macro to have a ceil-like function.
#define CEILING(x, y) (((x) + (y)-1) / (y))

// Handy macro for debug prints.
#if !defined NDEBUG || defined DEBUG_TEST
#define DEBUG_TEST 1
#else
#define DEBUG_TEST 0
#endif
#define DEBUG_PRINT(fmt, ...)                              \
    do {                                                   \
        if (DEBUG_TEST) fprintf(stderr, fmt, __VA_ARGS__); \
    } while (0)

// A password put into a union. This is easier to use with malloc and crypto
// functions.
typedef union {
    uint8_t bytes[PASSWORD_LENGTH];
    uint32_t i[CEILING(PASSWORD_LENGTH, 4)];
} Password;

// A digest put into a union.
typedef union {
    uint8_t bytes[HASH_LENGTH];
    uint32_t i[CEILING(HASH_LENGTH, 4)];
    uint64_t value;
} Digest;

/**
 * Prints a hash properly in the console.
 * @param digest the hash to print.
 */
void print_hash(const unsigned char *digest);

/**
 * Searches for a specific char array in a list of endpoints.
 * @param endpoints the list of endpoints (as a char array array).
 * @param plain_text the endpoint to find.
 * @param mt the number of endpoints in the list.
 * @param pwd_length the length of the password
 * @return the index of found value in the list if found, -1 otherwise.
 */
unsigned long search_endpoint(char **endpoints, char *plain_text, int mt, int pwd_length);

/**
 * Transforms a char array to a password.
 * @param text the char array.
 * @param password the resulting password.
 * @param pwd_length the length of the password.
 */
void char_to_password(char text[], Password *password, int pwd_length);

/**
 * Transforms a password to a char array.
 * @param password the password.
 * @param text the resulting chay array.
 * @param pwd_length the length of the password.
 */
void password_to_char(Password *password, char text[], int pwd_length);

/**
 * Transforms a char array into a digest.
 * @param text the char array.
 * @param digest the resulting digest.
 */
void char_to_digest(char text[], Digest *digest);

/**
 * Prints a digest in the console.
 * @param digest the digest to print.
 */
void display_digest(Digest *digest);

/**
 * Prints a password in the console.
 * @param pwd the password to print.
 */
void display_password(Password *pwd);

/**
 * Reduces a digest into a password.
 * @param char_digest the digest to reduce.
 * @param index the column index (using rainbow table means the reduction function depends on the column).
 * @param char_plain the result of the reduction.
 * @param pwd_length the length of the password to produce.
 */
void reduce_digest_old(char *char_digest, unsigned int index, char *char_plain, int pwd_length);

/**
 * Hashes a key into its NTLM digest.
 * @param key the char array to hash.
 * @param hash the NTLM hash of key.
 */
void ntlm(char *key, char *hash, int pwd_length);

/**
 * Perform the online attack with the startpoint and endpoint files.
 * @param start_path the path to the startpoint file.
 * @param end_path the path to the endpoint file.
 * @param digest the digest we're looking to crack.
 * @param password if found, the password corresponding to the digest.
 * @param pwd_length the password length, read in the files.
 */
void online_from_files(char *start_path, char *end_path, unsigned char *digest, char *password, int pwd_length);

int online_from_files_coverage(char *start_path, char *end_path, int pwd_length);


#endif