#include <cuda_stdint.h>

// The length of a NTLM hash.
#define HASH_LENGTH 16

// The length of a password.
#define PASSWORD_LENGTH 7

// The number of password to hash. (2^27)
#define PASSWORD_NUMBER 134217728

// A macro to have a ceil-like function.
#define CEILING(x, y) (((x) + (y)-1) / (y))

// A password put into an union. This is easier to use with mallocs and crypto
// functions.
typedef union {
    uint8_t bytes[PASSWORD_LENGTH];
    uint32_t i[CEILING(PASSWORD_LENGTH, 4)];
} Password;

// A digest put into an union.
typedef union {
    uint8_t bytes[HASH_LENGTH];
    uint32_t i[CEILING(HASH_LENGTH, 4)];
} Digest;

// Md4 context.
typedef struct md4_ctx_vector {
    uint32_t h[4];
    uint32_t w0[4];
    uint32_t w1[4];
    uint32_t w2[4];
    uint32_t w3[4];
    int len;
} md4_ctx_vector_t;

// Md4 constants.
typedef enum md4_constants {
    MD4M_A = 0x67452301U,
    MD4M_B = 0xefcdab89U,
    MD4M_C = 0x98badcfeU,
    MD4M_D = 0x10325476U,

    MD4S00 = 3,
    MD4S01 = 7,
    MD4S02 = 11,
    MD4S03 = 19,
    MD4S10 = 3,
    MD4S11 = 5,
    MD4S12 = 9,
    MD4S13 = 13,
    MD4S20 = 3,
    MD4S21 = 9,
    MD4S22 = 11,
    MD4S23 = 15,

    MD4C00 = 0x00000000U,
    MD4C01 = 0x5a827999U,
    MD4C02 = 0x6ed9eba1U

} md4_constants_t;

/*
    A NTLM kernel.
    This has been adapted from hashcat's m01000_a3-pure.cl module.
    https://github.com/hashcat/hashcat/blob/master/OpenCL/m01000_a3-pure.cl

    `passwords` is the array containing all passwords to hash.
    `digests` contains all hashed passwords.
*/
__global__ void ntlm(Password* passwords, Digest* digests);

__device__ uint32_t rotl32(const uint32_t a, const int n);
__device__ uint32_t rotr32(const uint32_t a, const int n);
__device__ uint32_t hc_bytealign(const uint32_t a, const uint32_t b,
                                 const int c);
__device__ void switch_buffer_by_offset_carry_le(uint32_t* w0, uint32_t* w1,
                                                 uint32_t* w2, uint32_t* w3,
                                                 uint32_t* c0, uint32_t* c1,
                                                 uint32_t* c2, uint32_t* c3,
                                                 const uint32_t offset);

__device__ void switch_buffer_by_offset_le(uint32_t* w0, uint32_t* w1,
                                           uint32_t* w2, uint32_t* w3,
                                           const uint32_t offset);
__device__ void make_utf16le(const uint32_t* in, uint32_t* out1,
                             uint32_t* out2);
__device__ void md4_init_vector(md4_ctx_vector_t* ctx);
__device__ void md4_transform_vector(const uint32_t* w0, const uint32_t* w1,
                                     uint32_t* w2, const uint32_t* w3,
                                     uint32_t* digest);
__device__ void md4_update_vector_64(md4_ctx_vector_t* ctx, uint32_t* w0,
                                     uint32_t* w1, uint32_t* w2, uint32_t* w3,
                                     const int len);
__device__ void md4_update_vector_utf16le(md4_ctx_vector_t* ctx,
                                          const uint32_t* w);
__device__ void append_helper_1x4(uint32_t* r, const uint32_t v,
                                  const uint32_t* m);

#define MD4_F_S(x, y, z) (((x) & (y)) | ((~(x)) & (z)))
#define MD4_G_S(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define MD4_H_S(x, y, z) ((x) ^ (y) ^ (z))

#define MD4_F(x, y, z) (((x) & (y)) | ((~(x)) & (z)))
#define MD4_G(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define MD4_H(x, y, z) ((x) ^ (y) ^ (z))

#define MD4_Fo(x, y, z) (MD4_F((x), (y), (z)))
#define MD4_Go(x, y, z) (MD4_G((x), (y), (z)))

#define MD4_STEP(f, a, b, c, d, x, K, s) \
    {                                    \
        a += K;                          \
        a += x + f(b, c, d);             \
        a = rotl32(a, s);                \
    }

#define MD4_STEP0(f, a, b, c, d, K, s) \
    {                                  \
        a += K + f(b, c, d);           \
        a = rotl32(a, s);              \
    }
