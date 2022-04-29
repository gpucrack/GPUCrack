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
#define PASSWORD_LENGTH 3

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
} Digest;

/**
 * Searches for a specific char array in a list of endpoints.
 * Uses binary search algorithm.
 * @param endpoints the SORTED list of endpoints (as a char array array).
 * @param plain_text the endpoint to find.
 * @param mt the number of endpoints in the list.
 * @param pwd_length the length of the password
 * @return the index of found value in the list if found, -1 otherwise.
 */
unsigned long search_endpoint(char **endpoints, char *plain_text, unsigned long mt, int pwd_length);

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
void char_to_digest(char text[], Digest *digest, int len);

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
void reduce_digest(char *char_digest, unsigned int index, char *char_plain, int pwd_length);

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
 * @param nbTable the number of tables to be searched into.
 * @param debug if true, prints more detailed results.
 */
void online_from_files(char *path, char *digest, char *password, int pwd_length, int nbTable, int debug);

/**
 * Generates passwords in an exhaustive fashion and tries to crack them.
 * @param start_path the path to the startpoint file.
 * @param end_path the path to the endpoint file.
 * @param pwd_length  the length of every password the table contains.
 * @param nb_cover the number of passwords to be cracked.
 * @return the number of passwords that were successfully cracked.
 */
int online_from_files_coverage(char *start_path, char *end_path, int pwd_length, int nb_cover);

/**
 * Checks whether the correct number of arguments was given when the program was called.
 * @param argc the number of arguments.
 * @return 0 if the check was successful
 */
int checkArgs(int argc);

/**
 * Retrieves the number of tables provided and the length of its passwords.
 * Also checks if the table exists, and verifies that every start points file has its corresponding end points file.
 * @param path the path to the table.
 * @param nbTable the number of table found for specified path.
 * @param pwdLength the length of every password the table contains.
 * @return 0 if no error was encountered
 */
int checkTables(char *path, int *nbTable, int *pwdLength);