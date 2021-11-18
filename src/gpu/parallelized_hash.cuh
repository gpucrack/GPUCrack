//
// Created by mynder on 18/11/2021.
//

#ifndef CUDA_NAIVE_EXHAUSTIVE_SEARCH_PARALLELIZED_HASH_CUH
#define CUDA_NAIVE_EXHAUSTIVE_SEARCH_PARALLELIZED_HASH_CUH

#define PASSWORD_NUMBER 100000000
#define MAX_PASSWORD_LENGTH 7

// A password, put into a struct so it's easier with mallocs.
typedef struct {
    BYTE chars[MAX_PASSWORD_LENGTH + 1];
} Password;

// A digest put into a struct.
typedef struct {
    BYTE bytes[MD5_BLOCK_SIZE];
} Digest;

#endif //CUDA_NAIVE_EXHAUSTIVE_SEARCH_PARALLELIZED_HASH_CUH
