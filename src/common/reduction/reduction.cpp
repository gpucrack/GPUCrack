#include "reduction.h"

// The character set used for passwords.
static const unsigned char charset[CHARSET_LENGTH] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
                                                      'D', 'E',
                                                      'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                                                      'S', 'T',
                                                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                                                      'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                                      'q', 'r',
                                                      's', 't',
                                                      'u', 'v', 'w', 'x', 'y', 'z', '-', '_'};
// The character set used for digests (NTLM hashes).
static const unsigned char hashset[DIGEST_CHARSET_LENGTH] = {0x88, 0x46, 0xF7, 0xEA, 0xEE, 0x8F, 0xB1,
                                                             0x17, 0xAD, 0x06, 0xBD, 0xD8, 0x30, 0xB7,
                                                             0x58, 0x6C};

void generate_pwd_from_text(char text[], Password *password) {
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        password->bytes[i] = text[i];
    }
}

void display_password(Password &pwd, bool br = true) {
    for (unsigned char i = 0; i < PASSWORD_LENGTH; i++) {
        printf("%c", (unsigned char) pwd.bytes[i]);
    }
    if (br) printf("\n");
}

void display_passwords(Password **passwords, int n) {
    for (int j = 0; j < n; j++) {
        display_password((*passwords)[j]);
    }
}

void display_digest(Digest &digest, bool br = true) {
    for (unsigned char i = 0; i < HASH_LENGTH; i++) {
        printf("%02X", (unsigned char) digest.bytes[i]);
    }
    if (br) printf("\n");
}

void display_digests(Digest **digests) {
    for (int j = 0; j < DEFAULT_PASSWORD_NUMBER; j++) {
        display_digest((*digests)[j]);
    }
}

void generate_password(unsigned long counter, Password &plain_text) {
    for (int i = PASSWORD_LENGTH - 1; i >= 0; i--) {
        plain_text.bytes[i] = charset[counter % CHARSET_LENGTH];
        counter /= CHARSET_LENGTH;
    }
}

void generate_passwords(Password **passwords, int n, unsigned long long h = 0) {
    for (int j = h; j < n + h; j++) {
        generate_password(j, (*passwords)[j - h]);
    }
}

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

void generate_digests_random(Digest **digests, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = HASH_LENGTH - 1; i >= 0; i--) {
            (*digests)[j].bytes[i] = hashset[rand() % CHARSET_LENGTH];
        }
    }
}

void generate_digests(Digest **digests, int n) {
    for (int j = 0; j < n; j++) {
        generate_digest(j, (*digests)[j]);
    }
}

void generate_digests_inverse(Digest **digests, int n) {
    for (int j = 0; j < n; j++) {
        generate_digest_inverse(j, (*digests)[j]);
    }
}

void reduce_digest_hellman(Digest &digest, Password &plain_text) {
    for (int i = 0; i < PASSWORD_LENGTH - 1; i++) {
        plain_text.bytes[i] = charset[(digest.bytes[i]) % CHARSET_LENGTH];
    }
}

void reduce_digest(unsigned long index, Digest &digest, Password &plain_text) {
    for (int i = 0; i < PASSWORD_LENGTH - 1; i++) {
        plain_text.bytes[i] = charset[(digest.bytes[i] + index) % CHARSET_LENGTH];
    }
}

void reduceDigest(unsigned int index, Digest * digest, Password * plain_text) {
    (*plain_text).i[0] =
            charset[((*digest).bytes[0] + index) % CHARSET_LENGTH] |
            (charset[((*digest).bytes[1] + index) % CHARSET_LENGTH] << 8)|
            (charset[((*digest).bytes[2] + index) % CHARSET_LENGTH] << 16)|
            (charset[((*digest).bytes[3] + index) % CHARSET_LENGTH] << 24);
    (*plain_text).i[1] =
            charset[((*digest).bytes[4] + index) % CHARSET_LENGTH] |
            (charset[((*digest).bytes[5] + index) % CHARSET_LENGTH] << 8)|
            (charset[((*digest).bytes[6] + index) % CHARSET_LENGTH] << 16)|
            (charset[((*digest).bytes[7] + index) % CHARSET_LENGTH] << 24);
}

// Does not work: 27 MR/s and 99.62% of duplicates in 10.000 reductions.
void reduce_digest_2(unsigned long index, Digest &digest, Password &plain_text) {
    uint8_t counter = digest.bytes[7];
    for (char i = 6; i >= 0; i--) {
        counter <<= 7;
        counter |= digest.bytes[i];
    }
    for (int i = PASSWORD_LENGTH - 1; i >= 0; i--) {
        plain_text.bytes[i] = charset[(counter + index) % CHARSET_LENGTH];
        counter /= 64;
    }
}

// Just for testing speed
void reduce_digest_3(Password &plain_text) {
    for (int i = 0; i < PASSWORD_LENGTH - 1; i++) {
        plain_text.bytes[i] = (uint8_t) 'a';
    }
}

void reduce_digests(Digest **digests, Password **plain_texts) {
    for (int j = 0; j < DEFAULT_PASSWORD_NUMBER; j++) {
        reduceDigest(j, (*digests)[j], (*plain_texts)[j]);
    }
}

inline int pwdcmp(Password &p1, Password &p2) {
    for (int i = 0; i < CEILING(PASSWORD_LENGTH, 4); i++) {
        if (p1.i[i] != p2.i[i]) {
            return false;
        }
    }
    return true;
}

int count_duplicates(Password **passwords, bool debug = false) {
    int count = 0;
    for (int i = 0; i < DEFAULT_PASSWORD_NUMBER; i++) {
        if (debug) printf("Searching for duplicate of password number %d...\n", i);
        for (int j = i + 1; j < DEFAULT_PASSWORD_NUMBER; j++) {
            // Increment count by 1 if duplicate found
            if (pwdcmp((*passwords)[i], (*passwords)[j])) {
                if (debug) {
                    printf("Found a duplicate : ");
                    display_password((*passwords)[i]);
                }
                count++;
                break;
            }
        }
    }
    return count;
}

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
/*int main() {

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

    int dup = count_duplicates(&passwords);
    printf("Found %d duplicate(s) among the %d reduced passwords (%f percent).\n", dup, DEFAULT_PASSWORD_NUMBER,
           ((double) dup / DEFAULT_PASSWORD_NUMBER) * 100);

    //display_passwords(&passwords);
    return 0;
}*/

void printbits(unsigned char v) {
    int i; // for C89 compatability
    for (i = 7; i >= 0; i--) putchar('0' + ((v >> i) & 1));
}

void print_pwd_as_binary(Password &password) {
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        printbits(password.bytes[i]);
        printf(" ");
    }
}

void print_pwds_as_binary(Password **passwords) {
    for (int i = 0; i < DEFAULT_PASSWORD_NUMBER; i++) {
        print_pwd_as_binary((*passwords)[i]);
        printf("\n");
    }
}


int main() {

    int n = 7 * 8;
    int p = 12;
    int s = n - p;

    // Initialize and allocate memory for a password array
    Password *start_p = NULL;
    Password *end_p = NULL;
    start_p = (Password *) malloc(sizeof(Password) * DEFAULT_PASSWORD_NUMBER);
    end_p = (Password *) malloc(sizeof(Password) * DEFAULT_PASSWORD_NUMBER);

    // Generate DEFAULT_PASSWORD_NUMBER digests
    printf("Generating start and end points...\n");
    generate_passwords(&start_p, DEFAULT_PASSWORD_NUMBER, 0);
    generate_passwords(&end_p, DEFAULT_PASSWORD_NUMBER, 2100000000);
    printf("Generation done!\n");

    display_passwords(&start_p, DEFAULT_PASSWORD_NUMBER);
    display_passwords(&end_p, DEFAULT_PASSWORD_NUMBER);

    print_pwds_as_binary(&start_p);
    print_pwds_as_binary(&end_p);

    printf("\n");

    // Start the chronometer...
    clock_t t;
    t = clock();

    // End the chronometer!
    t = clock() - t;
    double time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds
    double reduce_rate = (DEFAULT_PASSWORD_NUMBER / 1000000) / time_taken;

    printf("Storage of %d points ended after %f seconds.\n%f MR/s.\n", DEFAULT_PASSWORD_NUMBER,
           time_taken, reduce_rate);


    return 0;
}