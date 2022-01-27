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

// How many thread per block to launch the kernel
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

// A digest put into a union.
typedef union {
    uint8_t bytes[HASH_LENGTH];
    uint32_t i[CEILING(HASH_LENGTH, 4)];
} Digest;

/*
 * Generates a password given a char array.
 * text: literal to be put into the password
 * password: result
*/
void generate_pwd_from_text(char text[], Password *password);

/*
 * Displays a single password properly, char by char.
 * pwd: the password to display.
 */
void display_password(Password &pwd, bool br);

/*
 * Displays a password array properly with chars.
 * passwords: the password array to display.
 */
void display_passwords(Password **passwords);

/*
 * Displays a single digest properly.
 * digest: the digest to display.
 */
void display_digest(Digest &digest, bool br);

/*
 * Displays a digest array properly with chars.
 * digests: the digest array to display.
 */
void display_digests(Digest **digests);

/*
 * Generates a password corresponding to a given number.
 * Used to create the start points of the rainbow table.
 * counter: number corresponding to the chain's index
 * plain_text: password corresponding to its counter
*/
void generate_password(unsigned long counter, Password &plain_text);

/*
 * Fills passwords with n generated passwords.
 * passwords: the password array to fill.
 * n: the number of passwords to be generated.
 */
void generate_passwords(Password **passwords, int n);

/*
 * Generates a digest, filling it from right to left.
 * counter: this of it as a seed
 * hash: result digest
*/
void generate_digest(unsigned long counter, Digest &hash);

/*
 * Generates a digest, filling it from left to right.
 * counter: this of it as a seed
 * hash: result digest
*/
void generate_digest_inverse(unsigned long counter, Digest &hash);

/*
 * Fills digests with n pseudo-randomly generated digests.
 * digests: the digest array to fill.
 * n: the number of digests to be generated.
 */
void generate_digests_random(Digest **digests, int n);

/*
 * Fills digests with n incrementally generated digests.
 * 88888888, 88888846, 888888F7, 888888EA...
 * digests: the digest array to fill.
 * n: the number of digests to be generated.
 */
void generate_digests(Digest **digests, int n);


/*
 * Fills digests with n incrementally inverted generated digests.
 * 88888888, 46888888, F7888888, EA888888...
 * digests: the digest array to fill.
 * n: the number of digests to be generated.
 */
void generate_digests_inverse(Digest **digests, int n);

/*
 * Reduces a digest into a plain text like in Hellman tables, thus not using the column index.
 * This reduction function yields 0.031 % of duplicates on 100.000
 * digest: the digest to reduce
 * plain_text: the generated reduction
 */
void reduce_digest_hellman(Digest &digest, Password &plain_text);

/*
 * Reduces a digest into a plain text using the column index.
 * index: column index
 * digest: the digest to reduce
 * plain_text: the generated reduction
 */
void reduce_digest(unsigned long index, Digest &digest, Password &plain_text);

/*
 * Reduces every digest of an array into plain texts.
 * digests: the digest array to reduce
 * plain_texts: the generated reductions
 */
void reduce_digests(Digest **digests, Password **plain_texts);

/*
 * Compares two passwords.
 * return true if they are equal, false otherwise.
 */
inline int pwdcmp(Password &p1, Password &p2);

/*
 * Finds the number of duplicates in a password array
 */
int count_duplicates(Password **passwords, bool debug);

/*
 * Displays a reduction as "DIGEST --> PASSWORD"
 * digests: the array of digests
 * passwords: the array of passwords
 * n (optional): only display the n first reductions
 */
void display_reductions(Digest **digests, Password **passwords, int n);