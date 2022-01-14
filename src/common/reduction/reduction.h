#include <stdio.h>
#include <stdlib.h>

#define PLAIN_LENGTH 8  // the length of the generated plain text
#define CHARSET_LENGTH 62   // the number of characters in the charset

// Reduces a hash into a plain text of length PLAIN_LENGTH.
void reduce(unsigned long int index, const char *hash, char *plain);
