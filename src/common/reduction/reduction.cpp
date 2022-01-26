#include "reduction.h"

// The character set used for passwords.
static const char *charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";
// The character set used for digests (NTLM hashes).
static const char hashset[16] = {0x88, 0x46, 0xF7, 0xEA, 0xEE, 0x8F, 0xB1, 0x17, 0xAD, 0x06, 0xBD, 0xD8, 0x30, 0xB7,
                                 0x58, 0x6C};

/*
 * Generates a password given a char array.
 * text: literal to be put into the password
 * password: result
*/
void generate_pwd_from_text(char text[], Password *password) {
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        password->bytes[i] = text[i];
    }
}

/*
 * Displays a single password properly, char by char.
 * pwd: the password to display.
 */
void display_password(Password &pwd, bool br = true) {
    for (unsigned char byte: pwd.bytes) {
        printf("%c", (char) byte);
    }
    if (br) printf("\n");
}

/*
 * Displays a password array properly with chars.
 * passwords: the password array to display.
 */
void display_passwords(Password **passwords) {
    for (int j = 0; j < DEFAULT_PASSWORD_NUMBER; j++) {
        display_password((*passwords)[j]);
    }
}

/*
 * Displays a single digest properly.
 * digest: the digest to display.
 */
void display_digest(Digest &digest, bool br = true) {
    for (unsigned char byte: digest.bytes) {
        printf("%02X", byte);
    }
    if (br) printf("\n");
}


/*
 * Displays a digest array properly with chars.
 * digests: the digest array to display.
 */
void display_digests(Digest **digests) {
    for (int j = 0; j < DEFAULT_PASSWORD_NUMBER; j++) {
        display_digest((*digests)[j]);
    }
}

/*
 * Generates a password corresponding to a given number.
 * Used to create the start points of the rainbow table.
 * counter: number corresponding to the chain's index
 * plain_text: password corresponding to its counter
*/
void generate_password(unsigned long counter, Password &plain_text) {
    for (int i = PASSWORD_LENGTH - 1; i >= 0; i--) {
        plain_text.bytes[i] = charset[counter % CHARSET_LENGTH];
        counter /= CHARSET_LENGTH;
    }
}

/*
 * Fills passwords with n generated passwords.
 * passwords: the password array to fill.
 * n: the number of passwords to be generated.
 */
void generate_passwords(Password **passwords, int n) {
    for (int j = 0; j < n; j++) {
        generate_password(j, (*passwords)[j]);
    }
}

/*
 * Generates a pseudo-random digest.
 * counter: this of it as a seed
 * hash: result digest
*/
void generate_digest(unsigned long counter, Digest &hash) {
    for (int i = HASH_LENGTH - 1; i >= 0; i--) {
        hash.bytes[i] = hashset[counter % DIGEST_CHARSET_LENGTH];
        counter /= DIGEST_CHARSET_LENGTH;
    }
}

void generate_digest_inverse(unsigned long counter, Digest &hash) {
    for (int i = 0; i < HASH_LENGTH - 1; i++) {
        hash.bytes[i] = hashset[counter % DIGEST_CHARSET_LENGTH];
        counter /= DIGEST_CHARSET_LENGTH;
    }
}

/*
 * Fills digests with n pseudo-randomly generated digests.
 * digests: the digest array to fill.
 * n: the number of digests to be generated.
 */
void generate_digests_random(Digest **digests, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = HASH_LENGTH - 1; i >= 0; i--) {
            (*digests)[j].bytes[i] = hashset[rand() % CHARSET_LENGTH];
        }
    }
}

/*
 * Fills digests with n incrementally generated digests.
 * 88888888, 88888846, 888888F7, 888888EA...
 * digests: the digest array to fill.
 * n: the number of digests to be generated.
 */
void generate_digests(Digest **digests, int n) {
    for (int j = 0; j < n; j++) {
        generate_digest(j, (*digests)[j]);
    }
}

/*
 * Fills digests with n incrementally inverted generated digests.
 * 88888888, 46888888, F7888888, EA888888...
 * digests: the digest array to fill.
 * n: the number of digests to be generated.
 */
void generate_digests_inverse(Digest **digests, int n) {
    for (int j = 0; j < n; j++) {
        generate_digest_inverse(j, (*digests)[j]);
    }
}

/*
 * Reduces a digest into a plain text like in Hellman tables, thus not using the column index.
 * This reduction function yields 0.031 % of duplicates on 100.000
 * digest: the digest to reduce
 * plain_text: the generated reduction
 */
void reduce_digest_hellman(unsigned long index, Digest &digest, Password &plain_text) {
    for (int i = 0; i < PASSWORD_LENGTH - 1; i++) {
        plain_text.bytes[i] = charset[(digest.bytes[i]) % CHARSET_LENGTH];
    }
}

/*
 * Reduces a digest into a plain text using the column index.
 * index: column index
 * digest: the digest to reduce
 * plain_text: the generated reduction
 */
void reduce_digest(unsigned long index, Digest &digest, Password &plain_text) {
    for (int i = 0; i < PASSWORD_LENGTH - 1; i++) {
        plain_text.bytes[i] = charset[(digest.bytes[i] + index) % CHARSET_LENGTH];
    }
}

/*
 * Reduces every digest of an array into plain texts.
 * digests: the digest array to reduce
 * plain_texts: the generated reductions
 */
void reduce_digests(Digest **digests, Password **plain_texts) {
    for (int j = 0; j < DEFAULT_PASSWORD_NUMBER; j++) {
        reduce_digest(j, (*digests)[j], (*plain_texts)[j]);
    }
}

/*
 * Compares two passwords.
 * return true if they are equal, false otherwise.
 */
inline int pwdcmp(Password &p1, Password &p2) {
    for (int i = 0; i < CEILING(PASSWORD_LENGTH, 4); i++) {
        if (p1.i[i] != p2.i[i]) {
            return false;
        }
    }
    return true;
}

/*
 * Finds the number of duplicates in a password array
 */
int count_duplicates(Password **passwords, bool debug = false) {
    int count = 0;
    for (int i = 0; i < DEFAULT_PASSWORD_NUMBER; i++) {
        if (debug) printf("Searching for duplicate of password number %d...\n", i);
        for (int j = i + 1; j < DEFAULT_PASSWORD_NUMBER; j++) {
            // Increment count by 1 if duplicate found
            if (pwdcmp((*passwords)[i], (*passwords)[j])) {
                printf("Found a duplicate : ");
                display_password((*passwords)[i]);
                count++;
            }
        }
    }
    return count;
}

/*
 * Displays a reduction as "DIGEST --> PASSWORD"
 * digests: the array of digests
 * passwords: the array of passwords
 * n (optional): only display the n first recutions
 */
void display_reductions(Digest **digests, Password **passwords, int n = DEFAULT_PASSWORD_NUMBER) {
    for (int i = 0; i < n; i++) {
        display_digest((*digests)[i], false);
        printf(" --> ");
        display_password((*passwords)[i], false);
        printf("\n");
    }
}

/*
 * Tests the reduction speed and searches for duplicates in reduced hashes.
 */
int main() {

    // Initialize and allocate memory for a password array
    Password *passwords = NULL;
    passwords = (Password *) malloc(sizeof(Password) * DEFAULT_PASSWORD_NUMBER);

    // Initialize and allocate memory for a digest array
    Digest *digests = NULL;
    digests = (Digest *) malloc(sizeof(Digest) * DEFAULT_PASSWORD_NUMBER);

    // Generate DEFAULT_PASSWORD_NUMBER digests
    printf("Generating digests...\n");
    generate_digests_random(&digests, DEFAULT_PASSWORD_NUMBER);
    //display_digests(&digests);
    printf("Digest generation done!\n\nEngaging reduction...\n");

    // Start the chronometer...
    clock_t t;
    t = clock();

    // Reduce all those digests into passwords
    reduce_digests(&digests, &passwords);

    // End the chronometer!
    t = clock() - t;
    double time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds
    double reduce_rate = (DEFAULT_PASSWORD_NUMBER / 1000000) / time_taken;

    printf("Reduction of %d digests ended after %f seconds.\nReduction rate: %f MR/s.\n", DEFAULT_PASSWORD_NUMBER,
           time_taken, reduce_rate);

    display_reductions(&digests, &passwords, 5);

    /*int dup = count_duplicates(&passwords);
    printf("Found %d duplicate(s) among the %d reduced passwords (%f percent).\n", dup, DEFAULT_PASSWORD_NUMBER,
           ((double) dup / DEFAULT_PASSWORD_NUMBER) * 100);*/

    //display_passwords(&passwords);
    return 0;
}