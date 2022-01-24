#include <stdio.h>
#include <stdlib.h>

#include "../constants.cuh"
#include "../hash/commons.cuh"
#include "../hash/parallelized_hash.cuh"

// Reduces a hash into a plain text of length PLAIN_LENGTH.
void reduce(unsigned long int index, const char *hash, char *plain);
__global__ void reduce_kernel(int index, Digest *digests, Password *passwords);