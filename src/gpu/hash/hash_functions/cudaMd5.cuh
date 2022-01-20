#ifndef MYMD5_CUH
#define MYMD5_CUH

#include <cstdio>
#include "../../constants.cuh"

typedef unsigned char BYTE;  // 8-bit byte
typedef unsigned int WORD;  // 32-bit word, change to "long" for 16-bit machines

typedef struct {
    BYTE data[64];
    WORD datalen;
    unsigned long long bitlen;
    WORD state[4];
} CUDA_MD5_CTX;

__device__ void cuda_md5_transform(CUDA_MD5_CTX *ctx, const BYTE data[]);

__device__ void cuda_md5_init(CUDA_MD5_CTX *ctx);

__device__ void cuda_md5_update(CUDA_MD5_CTX *ctx, const BYTE data[],
                                size_t len);

__device__ void cuda_md5_final(CUDA_MD5_CTX *ctx, BYTE hash[]);

__global__ void kernel_md5_hash(Password *indata, Digest *outdata);

#endif  // MYMD5_CUH