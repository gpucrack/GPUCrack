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
#include <unistd.h>

// The length of the charset.
#define CHARSET_LENGTH 62

unsigned char charset[CHARSET_LENGTH] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                                         'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                                         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                                         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
// The length of the digest produced by the hash function (NTLM).
#define PASSWORD_LENGTH 6

// The length of the digest produced by the hash function (NTLM).
#define HASH_LENGTH 16

// A password put into a union.
typedef struct {
    uint8_t bytes[PASSWORD_LENGTH];
} Password;

// A digest put into a union.
typedef union {
    uint8_t bytes[HASH_LENGTH];
    uint32_t i[HASH_LENGTH / 4];
    uint64_t value[HASH_LENGTH/8];
} Digest;

/**
 * Searches for a specific char array in a list of end points.
 * Uses a binary search algorithm, therefore the list of end points must be sorted.
 * @param endpoints the SORTED list of endpoints (as a char array array).
 * @param plainText the endpoint to find.
 * @param mt the number of endpoints in the list.
 * @param pwdLength the length of the password
 * @return the index of found value in the list if found, -1 otherwise.
 */
unsigned long search_endpoint(char **endpoints, char *plainText, unsigned long mt, int pwdLength);

/**
 * Transforms a char array to a Password union.
 * @param text the char array.
 * @param password the resulting password.
 * @param pwdLength the length of the password.
 */
void char_to_password(char text[], Password *password, int pwdLength);

/**
 * Transforms a Password union to a char array.
 * @param password the password.
 * @param text the resulting chay array.
 * @param pwdLength the length of the password.
 */
void password_to_char(Password *password, char text[], int pwdLength);

/**
 * Transforms a char array into a Digest union.
 * @param text the char array.
 * @param digest the resulting digest.
 */
void char_to_digest(char text[], Digest *digest, int len);

/**
 * Reduces the given digest into a plain-text password using characters from the charset.
 * @param charDigest the digest to reduce.
 * @param index the column index (using rainbow table means the reduction function depends on the column).
 * @param charPlain the result of the reduction.
 * @param pwdLength the length of the password to produce.
 */
void reduce_digest(char *charDigest, unsigned int index, char *charPlain, int pwdLength);

/**
 * Hashes a plain-text char array into its NTLM digest.
 * @param key the char array to hash.
 * @param hash the NTLM hash of key.
 */
void ntlm(char *key, char *hash, int pwdLength);

/**
 * Tries to crack a password using rainbow tables.
 * @param path the path to the table's file(s) (without '_start_N.bin').
 * @param digest the digest we're looking to crack.
 * @param password if found, the password corresponding to the digest.
 * @param pwdLength the password length, read in the files.
 * @param nbTable the number of tables to be searched into.
 * @param debug if true, prints more detailed results.
 */
void online_from_files(char *path, char *digest, char *password, int pwdLength, int nbTable, int debug);

/**
 * Retrieves the number of tables provided and the length of its passwords.
 * Also checks if the table exists, and verifies that every start points file has its corresponding end points file.
 * @param path the path to the table.
 * @param nbTable the number of table found for specified path.
 * @param pwdLength the length of every password the table contains.
 * @return 0 if no error was encountered
 */
int checkTables(char *path, int *nbTable, int *pwdLength);