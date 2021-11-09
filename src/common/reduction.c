#include <stdlib.h>
#include <stdio.h>
#include "reduction.h"

char* charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
unsigned int reduction_offset = 55;
unsigned int pos = 55;

// All the fuunctions below calculate a plaintext from a hash. The result is stored in the plain argument.
// First implementation. Barrel shifting and XOR.
void reduceV1(unsigned long int columnIndex, const char* hash, char* plain) {
	for (unsigned long int i = 0; i < PLAIN_LENGTH; i++, plain++, hash++, columnIndex>>=8)
		*plain = charset[(unsigned char)(*hash ^ columnIndex) % CHARSET_LENGTH];
}

// Second implementation. No barrel shifting for columnIndex and + (instead of XOR).
void reduceV2(unsigned long int columnIndex, const char* hash, char* plain) {
	for (unsigned long int i = 0; i < PLAIN_LENGTH; i++, plain++, hash++)
		*plain = charset[(unsigned char)(*hash + columnIndex) % CHARSET_LENGTH];
}

// Third implementation. No barrel shifting for columnIndex and XOR.
void reduceV3(unsigned long int columnIndex, const char* hash, char* plain) {
	for (unsigned long int i = 0; i < PLAIN_LENGTH; i++, plain++, hash++)
		*plain = charset[(unsigned char)(*hash ^ columnIndex) % CHARSET_LENGTH];
}

// Fourth implementation. Incrementation of columnIndex and +.
void reduceV4(unsigned long int columnIndex, const char* hash, char* plain) {
	for (unsigned long int i = 0; i < PLAIN_LENGTH; i++, plain++, hash++, columnIndex++)
		*plain = charset[(unsigned char)(*hash + columnIndex) % CHARSET_LENGTH];
}

// Fifth implementation. Incrementation of columnIndex and XOR.
void reduceV5(unsigned long int columnIndex, const char* hash, char* plain) {
	for (unsigned long int i = 0; i < PLAIN_LENGTH; i++, plain++, hash++, columnIndex++)
		*plain = charset[(unsigned char)(*hash ^ columnIndex) % CHARSET_LENGTH];
}


/* // RainbowCrackalack implementation
void reduceRainbowCrackalack(unsigned long int columnIndex, const char* hash, char* plain) {
    unsigned long index = hash_to_index(hash, *hash_len, reduction_offset, CHARSET_LENGTH, pos);
    index_to_plaintext(index, charset, CHARSET_LENGTH, PLAIN_LENGTH, PLAIN_LENGTH, plaintext_space_up_to_index, plain, PLAIN_LENGTH);
    
}


unsigned long hash_to_index(unsigned char *hash_value, unsigned int hash_len, unsigned int reduction_offset, unsigned long plaintext_space_total, unsigned int pos) {
  unsigned long ret = hash_value[7];
  ret <<= 8;
  ret |= hash_value[6];
  ret <<= 8;
  ret |= hash_value[5];
  ret <<= 8;
  ret |= hash_value[4];
  ret <<= 8;
  ret |= hash_value[3];
  ret <<= 8;
  ret |= hash_value[2];
  ret <<= 8;
  ret |= hash_value[1];
  ret <<= 8;
  ret |= hash_value[0];

  return (ret + reduction_offset + pos) % plaintext_space_total;
}

inline void index_to_plaintext(unsigned long index, char *charset, unsigned int charset_len, unsigned int plaintext_len_min, unsigned int plaintext_len_max, unsigned long *plaintext_space_up_to_index, unsigned char *plaintext, unsigned int *plaintext_len) {
  for (int i = plaintext_len_max - 1; i >= plaintext_len_min - 1; i--) {
    if (index >= plaintext_space_up_to_index[i]) {
      *plaintext_len = i + 1;
      break;
    }
  }

  unsigned long index_x = index - plaintext_space_up_to_index[*plaintext_len - 1];
  for (int i = *plaintext_len - 1; i >= 0; i--) {
    plaintext[i] = charset[index_x % charset_len];
    index_x = index_x / charset_len;
  }

  return;
} */