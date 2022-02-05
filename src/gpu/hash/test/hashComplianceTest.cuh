#ifndef GPU_CRACK_HASHCOMPLIANCETEST_CUH
#define GPU_CRACK_HASHCOMPLIANCETEST_CUH

#include <cstdio>
#include <cstdlib>
#include <cctype>

#include "../hash_functions/ntlm.cuh"
#include "../hash.cu"
#include "../hash_functions/cudaMd5.cuh"
#include "../hash_functions/classicMd5.cuh"

int compliance(int passwordNumber, Password * passwords, Digest * result, int numberOfPass,
               const unsigned char * referencePassword, unsigned char * referenceResult);

#endif //GPU_CRACK_HASHCOMPLIANCETEST_CUH
