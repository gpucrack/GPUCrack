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



void reduce_digest(Digest *digest, unsigned long iteration, unsigned char table_number, Password *plain_text) {
    // pseudo-random counter based on the hash
    unsigned long counter = digest->bytes[7];
    for (char i = 6; i >= 0; i--) {
        counter <<= 8;
        counter |= digest->bytes[i];
    }
}

/*
 * Generates a password given a char array.
 * text: literal to be put into the password
 * password: result
*/
void generate_pwd(char text[], Password *password) {
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        password->bytes[i] = text[i];
    }
}

/*
 * Displays a single password properly, char by char.
 * pwd: the password to display.
 */
void display_password(const Password *pwd) {
    for (unsigned char byte: pwd->bytes) {
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
        for (unsigned char byte: (*passwords)[j].bytes) {
            printf("%c", (char) byte);
        }
        printf("\n");
    }
}

/*
 * Generates a password corresponding to a given number.
 * Used to create the start points of the rainbow table.
 * counter: number corresponding to the chain's index
 * plain_text: password corresponding to its counter
*/
void generate_password(unsigned long counter, Password *plain_text) {
    for (int i = PASSWORD_LENGTH - 1; i >= 0; i--) {
        plain_text->bytes[i] = charset[counter % CHARSET_LENGTH];
        counter /= CHARSET_LENGTH;
    }
}

/*
 * Fills passwords with n generated passwords.
 * passwords: the password array to fill.
 * n: the number of passwords to be generated.
 */
void generate_passwords(Password **passwords, int n) {
    unsigned long counter;
    for (int j = 0; j < n; j++) {
        counter = j;
        for (int i = PASSWORD_LENGTH - 1; i >= 0; i--) {
            (*passwords)[j].bytes[i] = charset[counter % CHARSET_LENGTH];
            counter /= CHARSET_LENGTH;
        }
    }
}

/*
 * Tests the reduction and displays it in the console.
 */
int main() {

    // Initialize a password array
    Password *passwords = NULL;
    passwords = (Password *) malloc(sizeof(Password) * DEFAULT_PASSWORD_NUMBER);

    generate_passwords(&passwords, DEFAULT_PASSWORD_NUMBER);
    display_passwords(&passwords);

    return 0;
}