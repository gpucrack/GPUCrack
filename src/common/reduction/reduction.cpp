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
void display_password(Password &pwd) {
    for (unsigned char byte: pwd.bytes) {
        printf("%c", (char) byte);
    }
    printf("\n");
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
void display_digest(Digest &digest) {
    for (unsigned char byte: digest.bytes) {
        printf("%02X", byte);
    }
    printf("\n");
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

/*
 * Fills digests with n generated digests.
 * digests: the digest array to fill.
 * n: the number of digests to be generated.
 */
void generate_digests(Digest **digests, int n) {
    for (int j = 0; j < n; j++) {
        generate_digest(j, (*digests)[j]);
    }
}

/*
 * Reduces a digest into a plain text.
 * digest: the digest to reduce
 * plain_text: the generated reduction
 */
void reduce_digest(unsigned long index, Digest &digest, Password &plain_text) {
    unsigned long counter = digest.bytes[HASH_LENGTH - 1];
    for (char i = HASH_LENGTH; i >= 0; i--) {
        counter <<= 1;
    }
    generate_password(index + counter, plain_text);
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
 * Tests the reduction speed.
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
    generate_digests(&digests, DEFAULT_PASSWORD_NUMBER);
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

    printf("Reduction of %d digests ended after %f seconds.\nReduction rate: %f MR/s\n", DEFAULT_PASSWORD_NUMBER,
           time_taken, reduce_rate);

    //display_passwords(&passwords);
    return 0;
}