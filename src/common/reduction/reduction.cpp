#include "reduction.h"

// The character set used.
static const char *charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";

/*
 * Reduces a hash into a plain text of length PLAIN_LENGTH.
 * index: index of the column in the table
 * hash: cypher text to be reduced
 * plain: result of the reduction
 */
void reduce(unsigned long int index, const char *hash, char *plain) {
    for (unsigned long int i = 0; i < PASSWORD_LENGTH; i++, plain++, hash++)
        *plain = charset[(unsigned char) (*hash ^ index) % CHARSET_LENGTH];
}

// May be useless tbh
char char_in_range(unsigned char n) {
    return charset[n];
}

// Generate passwords
void create_startpoint(unsigned long counter, Password *plain_text) {
    for (int i = PASSWORD_LENGTH - 1; i >= 0; i--) {
        plain_text->bytes[i] = charset[counter % CHARSET_LENGTH];
        counter /= CHARSET_LENGTH;
    }
}

/* Generates a password given a char array.
* text: cypher text to be reduced
* plain: result of the reduction
*/
void generate_pwd(char text[], Password *password) {
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        password->bytes[i] = text[i];
    }
}

void reduce_digest(Digest *digest, unsigned long iteration, unsigned char table_number, Password *plain_text) {
    // pseudo-random counter based on the hash
    unsigned long counter = digest->bytes[7];
    for (char i = 6; i >= 0; i--) {
        counter <<= 8;
        counter |= digest->bytes[i];
    }
}

/*
 * Tests the reduction and displays it in the console.
 */
int main() {
    Password *pwd = NULL;
    pwd = (Password *) malloc(sizeof(Password));

    char test[] = "123456789";

    printf("%lu\n", sizeof test);

    generate_pwd(test, pwd);


    for (unsigned char byte: pwd->bytes) {
        printf("%c", (char) byte);
    }
    printf("\n");

    return 0;
}