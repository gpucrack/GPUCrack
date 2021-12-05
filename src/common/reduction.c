#include "reduction.h"
#include <stdio.h>
#include <stdlib.h>

char *charset =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
unsigned int reduction_offset = 55;
unsigned int pos = 55;

// All the fuunctions below calculate a plaintext from a hash. The result is
// stored in the plain argument. First implementation. Barrel shifting and XOR.
void reduceV1(unsigned long int columnIndex, const char *hash, char *plain) {
  for (unsigned long int i = 0; i < PLAIN_LENGTH;
       i++, plain++, hash++, columnIndex >>= 8)
    *plain = charset[(unsigned char)(*hash ^ columnIndex) % CHARSET_LENGTH];
}

// Second implementation. No barrel shifting for columnIndex and + (instead of
// XOR).
void reduceV2(unsigned long int columnIndex, const char *hash, char *plain) {
  for (unsigned long int i = 0; i < PLAIN_LENGTH; i++, plain++, hash++)
    *plain = charset[(unsigned char)(*hash + columnIndex) % CHARSET_LENGTH];
}

// Third implementation. No barrel shifting for columnIndex and XOR.
void reduceV3(unsigned long int columnIndex, const char *hash, char *plain) {
  for (unsigned long int i = 0; i < PLAIN_LENGTH; i++, plain++, hash++)
    *plain = charset[(unsigned char)(*hash ^ columnIndex) % CHARSET_LENGTH];
}

// Fourth implementation. Incrementation of columnIndex and +.
void reduceV4(unsigned long int columnIndex, const char *hash, char *plain) {
  for (unsigned long int i = 0; i < PLAIN_LENGTH;
       i++, plain++, hash++, columnIndex++)
    *plain = charset[(unsigned char)(*hash + columnIndex) % CHARSET_LENGTH];
}

// Fifth implementation. Incrementation of columnIndex and XOR.
void reduceV5(unsigned long int columnIndex, const char *hash, char *plain) {
  for (unsigned long int i = 0; i < PLAIN_LENGTH;
       i++, plain++, hash++, columnIndex++)
    *plain = charset[(unsigned char)(*hash ^ columnIndex) % CHARSET_LENGTH];
}
