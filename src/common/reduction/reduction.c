#include "reduction.h"

char *charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";   // the characters to be used in the generated plain text

/*
 * Reduces a hash into a plain text of length PLAIN_LENGTH.
 * index: index of the column in the table
 * hash: cypher text to be reduced
 * plain: result of the reduction
 */
void reduce(unsigned long int index, const char *hash, char *plain) {
    for (unsigned long int i = 0; i < PLAIN_LENGTH; i++, plain++, hash++)
        *plain = charset[(unsigned char) (*hash ^ index) % CHARSET_LENGTH];
}

/*
 * Tests the reduction and displays it in the console.
 */
int main() {
    unsigned int index = 3;
    unsigned int index2 = 4;

    char *hash = "8846F7EAEE8FB117AD06BDD830B7586C";
    char *hash2 = "878D8014606CDA29677A44EFA1353FC7";
    char *plain = malloc(sizeof(char) * (PLAIN_LENGTH));
    char *plain2 = malloc(sizeof(char) * (PLAIN_LENGTH));

    reduce(index, hash, plain);
    reduce(index2, hash2, plain2);

    printf("%s ---> %s\n", hash, plain);
    printf("%s ---> %s\n", hash2, plain2);

    return 0;
}