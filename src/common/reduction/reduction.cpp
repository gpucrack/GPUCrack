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

/*
 * Generates a password corresponding to a given number.
 * Used to create the start points of the rainbow table.
 * counter: number corresponding to the chain's index
 * plain_text: password corresponding to its counter
*/
void create_startpoint(unsigned long counter, Password *plain_text) {
    for (int i = PASSWORD_LENGTH - 1; i >= 0; i--) {
        plain_text->bytes[i] = charset[counter % CHARSET_LENGTH];
        counter /= CHARSET_LENGTH;
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

void reduce_digest(Digest *digest, unsigned long iteration, unsigned char table_number, Password *plain_text) {
    // pseudo-random counter based on the hash
    unsigned long counter = digest->bytes[7];
    for (char i = 6; i >= 0; i--) {
        counter <<= 8;
        counter |= digest->bytes[i];
    }
}

/*
 * Displays a password properly, char by char.
 * pwd: the password to display.
 */
void display_password(const Password *pwd) {
    for (unsigned char byte: pwd->bytes) {
        printf("%c", (char) byte);
    }
    printf("\n");
}

/*
 * Tests the reduction and displays it in the console.
 */
int main() {
    Password *pwd = NULL;
    pwd = (Password *) malloc(sizeof(Password));

    // Generate one billion passwords
    for(unsigned long i = 0; i<DEFAULT_PASSWORD_NUMBER; i++) {

    }

    create_startpoint(1, pwd);

    display_password(pwd);

    return 0;
}