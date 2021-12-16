#include "ntlm.cuh"

__device__ uint32_t rotl32(const uint32_t a, const int n) {
    return ((a << n) | ((a >> (32 - n))));
}

__device__ uint32_t rotr32(const uint32_t a, const int n) {
    return ((a >> n) | ((a << (32 - n))));
}

__device__ uint32_t hc_byte_perm(const uint32_t a, const uint32_t b,
                                 const int c) {
    uint32_t r = 0;
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ uint32_t hc_bytealign(const uint32_t a, const uint32_t b,
                                 const int c) {
    const int c_mod_4 = c & 3;
    const int c_minus_4 = 4 - c_mod_4;
    const uint32_t r =
        hc_byte_perm(a, b, (0x76543210 >> (c_minus_4 * 4)) & 0xffff);

    return r;
}

__device__ void switch_buffer_by_offset_carry_le(uint32_t* w0, uint32_t* w1,
                                                 uint32_t* w2, uint32_t* w3,
                                                 uint32_t* c0, uint32_t* c1,
                                                 uint32_t* c2, uint32_t* c3,
                                                 const uint32_t offset) {
    const int offset_switch = offset / 4;
    switch (offset_switch) {
        case 0:
            c0[0] = hc_bytealign(w3[3], 0, offset);
            w3[3] = hc_bytealign(w3[2], w3[3], offset);
            w3[2] = hc_bytealign(w3[1], w3[2], offset);
            w3[1] = hc_bytealign(w3[0], w3[1], offset);
            w3[0] = hc_bytealign(w2[3], w3[0], offset);
            w2[3] = hc_bytealign(w2[2], w2[3], offset);
            w2[2] = hc_bytealign(w2[1], w2[2], offset);
            w2[1] = hc_bytealign(w2[0], w2[1], offset);
            w2[0] = hc_bytealign(w1[3], w2[0], offset);
            w1[3] = hc_bytealign(w1[2], w1[3], offset);
            w1[2] = hc_bytealign(w1[1], w1[2], offset);
            w1[1] = hc_bytealign(w1[0], w1[1], offset);
            w1[0] = hc_bytealign(w0[3], w1[0], offset);
            w0[3] = hc_bytealign(w0[2], w0[3], offset);
            w0[2] = hc_bytealign(w0[1], w0[2], offset);
            w0[1] = hc_bytealign(w0[0], w0[1], offset);
            w0[0] = hc_bytealign(0, w0[0], offset);

            break;

        case 1:
            c0[1] = hc_bytealign(w3[3], 0, offset);
            c0[0] = hc_bytealign(w3[2], w3[3], offset);
            w3[3] = hc_bytealign(w3[1], w3[2], offset);
            w3[2] = hc_bytealign(w3[0], w3[1], offset);
            w3[1] = hc_bytealign(w2[3], w3[0], offset);
            w3[0] = hc_bytealign(w2[2], w2[3], offset);
            w2[3] = hc_bytealign(w2[1], w2[2], offset);
            w2[2] = hc_bytealign(w2[0], w2[1], offset);
            w2[1] = hc_bytealign(w1[3], w2[0], offset);
            w2[0] = hc_bytealign(w1[2], w1[3], offset);
            w1[3] = hc_bytealign(w1[1], w1[2], offset);
            w1[2] = hc_bytealign(w1[0], w1[1], offset);
            w1[1] = hc_bytealign(w0[3], w1[0], offset);
            w1[0] = hc_bytealign(w0[2], w0[3], offset);
            w0[3] = hc_bytealign(w0[1], w0[2], offset);
            w0[2] = hc_bytealign(w0[0], w0[1], offset);
            w0[1] = hc_bytealign(0, w0[0], offset);
            w0[0] = 0;

            break;

        case 2:
            c0[2] = hc_bytealign(w3[3], 0, offset);
            c0[1] = hc_bytealign(w3[2], w3[3], offset);
            c0[0] = hc_bytealign(w3[1], w3[2], offset);
            w3[3] = hc_bytealign(w3[0], w3[1], offset);
            w3[2] = hc_bytealign(w2[3], w3[0], offset);
            w3[1] = hc_bytealign(w2[2], w2[3], offset);
            w3[0] = hc_bytealign(w2[1], w2[2], offset);
            w2[3] = hc_bytealign(w2[0], w2[1], offset);
            w2[2] = hc_bytealign(w1[3], w2[0], offset);
            w2[1] = hc_bytealign(w1[2], w1[3], offset);
            w2[0] = hc_bytealign(w1[1], w1[2], offset);
            w1[3] = hc_bytealign(w1[0], w1[1], offset);
            w1[2] = hc_bytealign(w0[3], w1[0], offset);
            w1[1] = hc_bytealign(w0[2], w0[3], offset);
            w1[0] = hc_bytealign(w0[1], w0[2], offset);
            w0[3] = hc_bytealign(w0[0], w0[1], offset);
            w0[2] = hc_bytealign(0, w0[0], offset);
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 3:
            c0[3] = hc_bytealign(w3[3], 0, offset);
            c0[2] = hc_bytealign(w3[2], w3[3], offset);
            c0[1] = hc_bytealign(w3[1], w3[2], offset);
            c0[0] = hc_bytealign(w3[0], w3[1], offset);
            w3[3] = hc_bytealign(w2[3], w3[0], offset);
            w3[2] = hc_bytealign(w2[2], w2[3], offset);
            w3[1] = hc_bytealign(w2[1], w2[2], offset);
            w3[0] = hc_bytealign(w2[0], w2[1], offset);
            w2[3] = hc_bytealign(w1[3], w2[0], offset);
            w2[2] = hc_bytealign(w1[2], w1[3], offset);
            w2[1] = hc_bytealign(w1[1], w1[2], offset);
            w2[0] = hc_bytealign(w1[0], w1[1], offset);
            w1[3] = hc_bytealign(w0[3], w1[0], offset);
            w1[2] = hc_bytealign(w0[2], w0[3], offset);
            w1[1] = hc_bytealign(w0[1], w0[2], offset);
            w1[0] = hc_bytealign(w0[0], w0[1], offset);
            w0[3] = hc_bytealign(0, w0[0], offset);
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 4:
            c1[0] = hc_bytealign(w3[3], 0, offset);
            c0[3] = hc_bytealign(w3[2], w3[3], offset);
            c0[2] = hc_bytealign(w3[1], w3[2], offset);
            c0[1] = hc_bytealign(w3[0], w3[1], offset);
            c0[0] = hc_bytealign(w2[3], w3[0], offset);
            w3[3] = hc_bytealign(w2[2], w2[3], offset);
            w3[2] = hc_bytealign(w2[1], w2[2], offset);
            w3[1] = hc_bytealign(w2[0], w2[1], offset);
            w3[0] = hc_bytealign(w1[3], w2[0], offset);
            w2[3] = hc_bytealign(w1[2], w1[3], offset);
            w2[2] = hc_bytealign(w1[1], w1[2], offset);
            w2[1] = hc_bytealign(w1[0], w1[1], offset);
            w2[0] = hc_bytealign(w0[3], w1[0], offset);
            w1[3] = hc_bytealign(w0[2], w0[3], offset);
            w1[2] = hc_bytealign(w0[1], w0[2], offset);
            w1[1] = hc_bytealign(w0[0], w0[1], offset);
            w1[0] = hc_bytealign(0, w0[0], offset);
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 5:
            c1[1] = hc_bytealign(w3[3], 0, offset);
            c1[0] = hc_bytealign(w3[2], w3[3], offset);
            c0[3] = hc_bytealign(w3[1], w3[2], offset);
            c0[2] = hc_bytealign(w3[0], w3[1], offset);
            c0[1] = hc_bytealign(w2[3], w3[0], offset);
            c0[0] = hc_bytealign(w2[2], w2[3], offset);
            w3[3] = hc_bytealign(w2[1], w2[2], offset);
            w3[2] = hc_bytealign(w2[0], w2[1], offset);
            w3[1] = hc_bytealign(w1[3], w2[0], offset);
            w3[0] = hc_bytealign(w1[2], w1[3], offset);
            w2[3] = hc_bytealign(w1[1], w1[2], offset);
            w2[2] = hc_bytealign(w1[0], w1[1], offset);
            w2[1] = hc_bytealign(w0[3], w1[0], offset);
            w2[0] = hc_bytealign(w0[2], w0[3], offset);
            w1[3] = hc_bytealign(w0[1], w0[2], offset);
            w1[2] = hc_bytealign(w0[0], w0[1], offset);
            w1[1] = hc_bytealign(0, w0[0], offset);
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 6:
            c1[2] = hc_bytealign(w3[3], 0, offset);
            c1[1] = hc_bytealign(w3[2], w3[3], offset);
            c1[0] = hc_bytealign(w3[1], w3[2], offset);
            c0[3] = hc_bytealign(w3[0], w3[1], offset);
            c0[2] = hc_bytealign(w2[3], w3[0], offset);
            c0[1] = hc_bytealign(w2[2], w2[3], offset);
            c0[0] = hc_bytealign(w2[1], w2[2], offset);
            w3[3] = hc_bytealign(w2[0], w2[1], offset);
            w3[2] = hc_bytealign(w1[3], w2[0], offset);
            w3[1] = hc_bytealign(w1[2], w1[3], offset);
            w3[0] = hc_bytealign(w1[1], w1[2], offset);
            w2[3] = hc_bytealign(w1[0], w1[1], offset);
            w2[2] = hc_bytealign(w0[3], w1[0], offset);
            w2[1] = hc_bytealign(w0[2], w0[3], offset);
            w2[0] = hc_bytealign(w0[1], w0[2], offset);
            w1[3] = hc_bytealign(w0[0], w0[1], offset);
            w1[2] = hc_bytealign(0, w0[0], offset);
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 7:
            c1[3] = hc_bytealign(w3[3], 0, offset);
            c1[2] = hc_bytealign(w3[2], w3[3], offset);
            c1[1] = hc_bytealign(w3[1], w3[2], offset);
            c1[0] = hc_bytealign(w3[0], w3[1], offset);
            c0[3] = hc_bytealign(w2[3], w3[0], offset);
            c0[2] = hc_bytealign(w2[2], w2[3], offset);
            c0[1] = hc_bytealign(w2[1], w2[2], offset);
            c0[0] = hc_bytealign(w2[0], w2[1], offset);
            w3[3] = hc_bytealign(w1[3], w2[0], offset);
            w3[2] = hc_bytealign(w1[2], w1[3], offset);
            w3[1] = hc_bytealign(w1[1], w1[2], offset);
            w3[0] = hc_bytealign(w1[0], w1[1], offset);
            w2[3] = hc_bytealign(w0[3], w1[0], offset);
            w2[2] = hc_bytealign(w0[2], w0[3], offset);
            w2[1] = hc_bytealign(w0[1], w0[2], offset);
            w2[0] = hc_bytealign(w0[0], w0[1], offset);
            w1[3] = hc_bytealign(0, w0[0], offset);
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 8:
            c2[0] = hc_bytealign(w3[3], 0, offset);
            c1[3] = hc_bytealign(w3[2], w3[3], offset);
            c1[2] = hc_bytealign(w3[1], w3[2], offset);
            c1[1] = hc_bytealign(w3[0], w3[1], offset);
            c1[0] = hc_bytealign(w2[3], w3[0], offset);
            c0[3] = hc_bytealign(w2[2], w2[3], offset);
            c0[2] = hc_bytealign(w2[1], w2[2], offset);
            c0[1] = hc_bytealign(w2[0], w2[1], offset);
            c0[0] = hc_bytealign(w1[3], w2[0], offset);
            w3[3] = hc_bytealign(w1[2], w1[3], offset);
            w3[2] = hc_bytealign(w1[1], w1[2], offset);
            w3[1] = hc_bytealign(w1[0], w1[1], offset);
            w3[0] = hc_bytealign(w0[3], w1[0], offset);
            w2[3] = hc_bytealign(w0[2], w0[3], offset);
            w2[2] = hc_bytealign(w0[1], w0[2], offset);
            w2[1] = hc_bytealign(w0[0], w0[1], offset);
            w2[0] = hc_bytealign(0, w0[0], offset);
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 9:
            c2[1] = hc_bytealign(w3[3], 0, offset);
            c2[0] = hc_bytealign(w3[2], w3[3], offset);
            c1[3] = hc_bytealign(w3[1], w3[2], offset);
            c1[2] = hc_bytealign(w3[0], w3[1], offset);
            c1[1] = hc_bytealign(w2[3], w3[0], offset);
            c1[0] = hc_bytealign(w2[2], w2[3], offset);
            c0[3] = hc_bytealign(w2[1], w2[2], offset);
            c0[2] = hc_bytealign(w2[0], w2[1], offset);
            c0[1] = hc_bytealign(w1[3], w2[0], offset);
            c0[0] = hc_bytealign(w1[2], w1[3], offset);
            w3[3] = hc_bytealign(w1[1], w1[2], offset);
            w3[2] = hc_bytealign(w1[0], w1[1], offset);
            w3[1] = hc_bytealign(w0[3], w1[0], offset);
            w3[0] = hc_bytealign(w0[2], w0[3], offset);
            w2[3] = hc_bytealign(w0[1], w0[2], offset);
            w2[2] = hc_bytealign(w0[0], w0[1], offset);
            w2[1] = hc_bytealign(0, w0[0], offset);
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 10:
            c2[2] = hc_bytealign(w3[3], 0, offset);
            c2[1] = hc_bytealign(w3[2], w3[3], offset);
            c2[0] = hc_bytealign(w3[1], w3[2], offset);
            c1[3] = hc_bytealign(w3[0], w3[1], offset);
            c1[2] = hc_bytealign(w2[3], w3[0], offset);
            c1[1] = hc_bytealign(w2[2], w2[3], offset);
            c1[0] = hc_bytealign(w2[1], w2[2], offset);
            c0[3] = hc_bytealign(w2[0], w2[1], offset);
            c0[2] = hc_bytealign(w1[3], w2[0], offset);
            c0[1] = hc_bytealign(w1[2], w1[3], offset);
            c0[0] = hc_bytealign(w1[1], w1[2], offset);
            w3[3] = hc_bytealign(w1[0], w1[1], offset);
            w3[2] = hc_bytealign(w0[3], w1[0], offset);
            w3[1] = hc_bytealign(w0[2], w0[3], offset);
            w3[0] = hc_bytealign(w0[1], w0[2], offset);
            w2[3] = hc_bytealign(w0[0], w0[1], offset);
            w2[2] = hc_bytealign(0, w0[0], offset);
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 11:
            c2[3] = hc_bytealign(w3[3], 0, offset);
            c2[2] = hc_bytealign(w3[2], w3[3], offset);
            c2[1] = hc_bytealign(w3[1], w3[2], offset);
            c2[0] = hc_bytealign(w3[0], w3[1], offset);
            c1[3] = hc_bytealign(w2[3], w3[0], offset);
            c1[2] = hc_bytealign(w2[2], w2[3], offset);
            c1[1] = hc_bytealign(w2[1], w2[2], offset);
            c1[0] = hc_bytealign(w2[0], w2[1], offset);
            c0[3] = hc_bytealign(w1[3], w2[0], offset);
            c0[2] = hc_bytealign(w1[2], w1[3], offset);
            c0[1] = hc_bytealign(w1[1], w1[2], offset);
            c0[0] = hc_bytealign(w1[0], w1[1], offset);
            w3[3] = hc_bytealign(w0[3], w1[0], offset);
            w3[2] = hc_bytealign(w0[2], w0[3], offset);
            w3[1] = hc_bytealign(w0[1], w0[2], offset);
            w3[0] = hc_bytealign(w0[0], w0[1], offset);
            w2[3] = hc_bytealign(0, w0[0], offset);
            w2[2] = 0;
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 12:
            c3[0] = hc_bytealign(w3[3], 0, offset);
            c2[3] = hc_bytealign(w3[2], w3[3], offset);
            c2[2] = hc_bytealign(w3[1], w3[2], offset);
            c2[1] = hc_bytealign(w3[0], w3[1], offset);
            c2[0] = hc_bytealign(w2[3], w3[0], offset);
            c1[3] = hc_bytealign(w2[2], w2[3], offset);
            c1[2] = hc_bytealign(w2[1], w2[2], offset);
            c1[1] = hc_bytealign(w2[0], w2[1], offset);
            c1[0] = hc_bytealign(w1[3], w2[0], offset);
            c0[3] = hc_bytealign(w1[2], w1[3], offset);
            c0[2] = hc_bytealign(w1[1], w1[2], offset);
            c0[1] = hc_bytealign(w1[0], w1[1], offset);
            c0[0] = hc_bytealign(w0[3], w1[0], offset);
            w3[3] = hc_bytealign(w0[2], w0[3], offset);
            w3[2] = hc_bytealign(w0[1], w0[2], offset);
            w3[1] = hc_bytealign(w0[0], w0[1], offset);
            w3[0] = hc_bytealign(0, w0[0], offset);
            w2[3] = 0;
            w2[2] = 0;
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 13:
            c3[1] = hc_bytealign(w3[3], 0, offset);
            c3[0] = hc_bytealign(w3[2], w3[3], offset);
            c2[3] = hc_bytealign(w3[1], w3[2], offset);
            c2[2] = hc_bytealign(w3[0], w3[1], offset);
            c2[1] = hc_bytealign(w2[3], w3[0], offset);
            c2[0] = hc_bytealign(w2[2], w2[3], offset);
            c1[3] = hc_bytealign(w2[1], w2[2], offset);
            c1[2] = hc_bytealign(w2[0], w2[1], offset);
            c1[1] = hc_bytealign(w1[3], w2[0], offset);
            c1[0] = hc_bytealign(w1[2], w1[3], offset);
            c0[3] = hc_bytealign(w1[1], w1[2], offset);
            c0[2] = hc_bytealign(w1[0], w1[1], offset);
            c0[1] = hc_bytealign(w0[3], w1[0], offset);
            c0[0] = hc_bytealign(w0[2], w0[3], offset);
            w3[3] = hc_bytealign(w0[1], w0[2], offset);
            w3[2] = hc_bytealign(w0[0], w0[1], offset);
            w3[1] = hc_bytealign(0, w0[0], offset);
            w3[0] = 0;
            w2[3] = 0;
            w2[2] = 0;
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 14:
            c3[2] = hc_bytealign(w3[3], 0, offset);
            c3[1] = hc_bytealign(w3[2], w3[3], offset);
            c3[0] = hc_bytealign(w3[1], w3[2], offset);
            c2[3] = hc_bytealign(w3[0], w3[1], offset);
            c2[2] = hc_bytealign(w2[3], w3[0], offset);
            c2[1] = hc_bytealign(w2[2], w2[3], offset);
            c2[0] = hc_bytealign(w2[1], w2[2], offset);
            c1[3] = hc_bytealign(w2[0], w2[1], offset);
            c1[2] = hc_bytealign(w1[3], w2[0], offset);
            c1[1] = hc_bytealign(w1[2], w1[3], offset);
            c1[0] = hc_bytealign(w1[1], w1[2], offset);
            c0[3] = hc_bytealign(w1[0], w1[1], offset);
            c0[2] = hc_bytealign(w0[3], w1[0], offset);
            c0[1] = hc_bytealign(w0[2], w0[3], offset);
            c0[0] = hc_bytealign(w0[1], w0[2], offset);
            w3[3] = hc_bytealign(w0[0], w0[1], offset);
            w3[2] = hc_bytealign(0, w0[0], offset);
            w3[1] = 0;
            w3[0] = 0;
            w2[3] = 0;
            w2[2] = 0;
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 15:
            c3[3] = hc_bytealign(w3[3], 0, offset);
            c3[2] = hc_bytealign(w3[2], w3[3], offset);
            c3[1] = hc_bytealign(w3[1], w3[2], offset);
            c3[0] = hc_bytealign(w3[0], w3[1], offset);
            c2[3] = hc_bytealign(w2[3], w3[0], offset);
            c2[2] = hc_bytealign(w2[2], w2[3], offset);
            c2[1] = hc_bytealign(w2[1], w2[2], offset);
            c2[0] = hc_bytealign(w2[0], w2[1], offset);
            c1[3] = hc_bytealign(w1[3], w2[0], offset);
            c1[2] = hc_bytealign(w1[2], w1[3], offset);
            c1[1] = hc_bytealign(w1[1], w1[2], offset);
            c1[0] = hc_bytealign(w1[0], w1[1], offset);
            c0[3] = hc_bytealign(w0[3], w1[0], offset);
            c0[2] = hc_bytealign(w0[2], w0[3], offset);
            c0[1] = hc_bytealign(w0[1], w0[2], offset);
            c0[0] = hc_bytealign(w0[0], w0[1], offset);
            w3[3] = hc_bytealign(0, w0[0], offset);
            w3[2] = 0;
            w3[1] = 0;
            w3[0] = 0;
            w2[3] = 0;
            w2[2] = 0;
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;
    }
}

__device__ void switch_buffer_by_offset_le(uint32_t* w0, uint32_t* w1,
                                           uint32_t* w2, uint32_t* w3,
                                           const uint32_t offset) {
    const int offset_switch = offset / 4;
    const int offset_mod_4 = offset & 3;
    const int offset_minus_4 = 4 - offset_mod_4;
    const int selector = (0x76543210 >> (offset_minus_4 * 4)) & 0xffff;

    switch (offset_switch) {
        case 0:
            w3[3] = hc_byte_perm(w3[2], w3[3], selector);
            w3[2] = hc_byte_perm(w3[1], w3[2], selector);
            w3[1] = hc_byte_perm(w3[0], w3[1], selector);
            w3[0] = hc_byte_perm(w2[3], w3[0], selector);
            w2[3] = hc_byte_perm(w2[2], w2[3], selector);
            w2[2] = hc_byte_perm(w2[1], w2[2], selector);
            w2[1] = hc_byte_perm(w2[0], w2[1], selector);
            w2[0] = hc_byte_perm(w1[3], w2[0], selector);
            w1[3] = hc_byte_perm(w1[2], w1[3], selector);
            w1[2] = hc_byte_perm(w1[1], w1[2], selector);
            w1[1] = hc_byte_perm(w1[0], w1[1], selector);
            w1[0] = hc_byte_perm(w0[3], w1[0], selector);
            w0[3] = hc_byte_perm(w0[2], w0[3], selector);
            w0[2] = hc_byte_perm(w0[1], w0[2], selector);
            w0[1] = hc_byte_perm(w0[0], w0[1], selector);
            w0[0] = hc_byte_perm(0, w0[0], selector);

            break;

        case 1:
            w3[3] = hc_byte_perm(w3[1], w3[2], selector);
            w3[2] = hc_byte_perm(w3[0], w3[1], selector);
            w3[1] = hc_byte_perm(w2[3], w3[0], selector);
            w3[0] = hc_byte_perm(w2[2], w2[3], selector);
            w2[3] = hc_byte_perm(w2[1], w2[2], selector);
            w2[2] = hc_byte_perm(w2[0], w2[1], selector);
            w2[1] = hc_byte_perm(w1[3], w2[0], selector);
            w2[0] = hc_byte_perm(w1[2], w1[3], selector);
            w1[3] = hc_byte_perm(w1[1], w1[2], selector);
            w1[2] = hc_byte_perm(w1[0], w1[1], selector);
            w1[1] = hc_byte_perm(w0[3], w1[0], selector);
            w1[0] = hc_byte_perm(w0[2], w0[3], selector);
            w0[3] = hc_byte_perm(w0[1], w0[2], selector);
            w0[2] = hc_byte_perm(w0[0], w0[1], selector);
            w0[1] = hc_byte_perm(0, w0[0], selector);
            w0[0] = 0;

            break;

        case 2:
            w3[3] = hc_byte_perm(w3[0], w3[1], selector);
            w3[2] = hc_byte_perm(w2[3], w3[0], selector);
            w3[1] = hc_byte_perm(w2[2], w2[3], selector);
            w3[0] = hc_byte_perm(w2[1], w2[2], selector);
            w2[3] = hc_byte_perm(w2[0], w2[1], selector);
            w2[2] = hc_byte_perm(w1[3], w2[0], selector);
            w2[1] = hc_byte_perm(w1[2], w1[3], selector);
            w2[0] = hc_byte_perm(w1[1], w1[2], selector);
            w1[3] = hc_byte_perm(w1[0], w1[1], selector);
            w1[2] = hc_byte_perm(w0[3], w1[0], selector);
            w1[1] = hc_byte_perm(w0[2], w0[3], selector);
            w1[0] = hc_byte_perm(w0[1], w0[2], selector);
            w0[3] = hc_byte_perm(w0[0], w0[1], selector);
            w0[2] = hc_byte_perm(0, w0[0], selector);
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 3:
            w3[3] = hc_byte_perm(w2[3], w3[0], selector);
            w3[2] = hc_byte_perm(w2[2], w2[3], selector);
            w3[1] = hc_byte_perm(w2[1], w2[2], selector);
            w3[0] = hc_byte_perm(w2[0], w2[1], selector);
            w2[3] = hc_byte_perm(w1[3], w2[0], selector);
            w2[2] = hc_byte_perm(w1[2], w1[3], selector);
            w2[1] = hc_byte_perm(w1[1], w1[2], selector);
            w2[0] = hc_byte_perm(w1[0], w1[1], selector);
            w1[3] = hc_byte_perm(w0[3], w1[0], selector);
            w1[2] = hc_byte_perm(w0[2], w0[3], selector);
            w1[1] = hc_byte_perm(w0[1], w0[2], selector);
            w1[0] = hc_byte_perm(w0[0], w0[1], selector);
            w0[3] = hc_byte_perm(0, w0[0], selector);
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 4:
            w3[3] = hc_byte_perm(w2[2], w2[3], selector);
            w3[2] = hc_byte_perm(w2[1], w2[2], selector);
            w3[1] = hc_byte_perm(w2[0], w2[1], selector);
            w3[0] = hc_byte_perm(w1[3], w2[0], selector);
            w2[3] = hc_byte_perm(w1[2], w1[3], selector);
            w2[2] = hc_byte_perm(w1[1], w1[2], selector);
            w2[1] = hc_byte_perm(w1[0], w1[1], selector);
            w2[0] = hc_byte_perm(w0[3], w1[0], selector);
            w1[3] = hc_byte_perm(w0[2], w0[3], selector);
            w1[2] = hc_byte_perm(w0[1], w0[2], selector);
            w1[1] = hc_byte_perm(w0[0], w0[1], selector);
            w1[0] = hc_byte_perm(0, w0[0], selector);
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 5:
            w3[3] = hc_byte_perm(w2[1], w2[2], selector);
            w3[2] = hc_byte_perm(w2[0], w2[1], selector);
            w3[1] = hc_byte_perm(w1[3], w2[0], selector);
            w3[0] = hc_byte_perm(w1[2], w1[3], selector);
            w2[3] = hc_byte_perm(w1[1], w1[2], selector);
            w2[2] = hc_byte_perm(w1[0], w1[1], selector);
            w2[1] = hc_byte_perm(w0[3], w1[0], selector);
            w2[0] = hc_byte_perm(w0[2], w0[3], selector);
            w1[3] = hc_byte_perm(w0[1], w0[2], selector);
            w1[2] = hc_byte_perm(w0[0], w0[1], selector);
            w1[1] = hc_byte_perm(0, w0[0], selector);
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 6:
            w3[3] = hc_byte_perm(w2[0], w2[1], selector);
            w3[2] = hc_byte_perm(w1[3], w2[0], selector);
            w3[1] = hc_byte_perm(w1[2], w1[3], selector);
            w3[0] = hc_byte_perm(w1[1], w1[2], selector);
            w2[3] = hc_byte_perm(w1[0], w1[1], selector);
            w2[2] = hc_byte_perm(w0[3], w1[0], selector);
            w2[1] = hc_byte_perm(w0[2], w0[3], selector);
            w2[0] = hc_byte_perm(w0[1], w0[2], selector);
            w1[3] = hc_byte_perm(w0[0], w0[1], selector);
            w1[2] = hc_byte_perm(0, w0[0], selector);
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 7:
            w3[3] = hc_byte_perm(w1[3], w2[0], selector);
            w3[2] = hc_byte_perm(w1[2], w1[3], selector);
            w3[1] = hc_byte_perm(w1[1], w1[2], selector);
            w3[0] = hc_byte_perm(w1[0], w1[1], selector);
            w2[3] = hc_byte_perm(w0[3], w1[0], selector);
            w2[2] = hc_byte_perm(w0[2], w0[3], selector);
            w2[1] = hc_byte_perm(w0[1], w0[2], selector);
            w2[0] = hc_byte_perm(w0[0], w0[1], selector);
            w1[3] = hc_byte_perm(0, w0[0], selector);
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 8:
            w3[3] = hc_byte_perm(w1[2], w1[3], selector);
            w3[2] = hc_byte_perm(w1[1], w1[2], selector);
            w3[1] = hc_byte_perm(w1[0], w1[1], selector);
            w3[0] = hc_byte_perm(w0[3], w1[0], selector);
            w2[3] = hc_byte_perm(w0[2], w0[3], selector);
            w2[2] = hc_byte_perm(w0[1], w0[2], selector);
            w2[1] = hc_byte_perm(w0[0], w0[1], selector);
            w2[0] = hc_byte_perm(0, w0[0], selector);
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 9:
            w3[3] = hc_byte_perm(w1[1], w1[2], selector);
            w3[2] = hc_byte_perm(w1[0], w1[1], selector);
            w3[1] = hc_byte_perm(w0[3], w1[0], selector);
            w3[0] = hc_byte_perm(w0[2], w0[3], selector);
            w2[3] = hc_byte_perm(w0[1], w0[2], selector);
            w2[2] = hc_byte_perm(w0[0], w0[1], selector);
            w2[1] = hc_byte_perm(0, w0[0], selector);
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 10:
            w3[3] = hc_byte_perm(w1[0], w1[1], selector);
            w3[2] = hc_byte_perm(w0[3], w1[0], selector);
            w3[1] = hc_byte_perm(w0[2], w0[3], selector);
            w3[0] = hc_byte_perm(w0[1], w0[2], selector);
            w2[3] = hc_byte_perm(w0[0], w0[1], selector);
            w2[2] = hc_byte_perm(0, w0[0], selector);
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 11:
            w3[3] = hc_byte_perm(w0[3], w1[0], selector);
            w3[2] = hc_byte_perm(w0[2], w0[3], selector);
            w3[1] = hc_byte_perm(w0[1], w0[2], selector);
            w3[0] = hc_byte_perm(w0[0], w0[1], selector);
            w2[3] = hc_byte_perm(0, w0[0], selector);
            w2[2] = 0;
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 12:
            w3[3] = hc_byte_perm(w0[2], w0[3], selector);
            w3[2] = hc_byte_perm(w0[1], w0[2], selector);
            w3[1] = hc_byte_perm(w0[0], w0[1], selector);
            w3[0] = hc_byte_perm(0, w0[0], selector);
            w2[3] = 0;
            w2[2] = 0;
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 13:
            w3[3] = hc_byte_perm(w0[1], w0[2], selector);
            w3[2] = hc_byte_perm(w0[0], w0[1], selector);
            w3[1] = hc_byte_perm(0, w0[0], selector);
            w3[0] = 0;
            w2[3] = 0;
            w2[2] = 0;
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 14:
            w3[3] = hc_byte_perm(w0[0], w0[1], selector);
            w3[2] = hc_byte_perm(0, w0[0], selector);
            w3[1] = 0;
            w3[0] = 0;
            w2[3] = 0;
            w2[2] = 0;
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;

        case 15:
            w3[3] = hc_byte_perm(0, w0[0], selector);
            w3[2] = 0;
            w3[1] = 0;
            w3[0] = 0;
            w2[3] = 0;
            w2[2] = 0;
            w2[1] = 0;
            w2[0] = 0;
            w1[3] = 0;
            w1[2] = 0;
            w1[1] = 0;
            w1[0] = 0;
            w0[3] = 0;
            w0[2] = 0;
            w0[1] = 0;
            w0[0] = 0;

            break;
    }
}

__device__ void make_utf16le(const uint32_t* in, uint32_t* out1,
                             uint32_t* out2) {
    out2[3] = hc_byte_perm(in[3], 0, 0x7372);
    out2[2] = hc_byte_perm(in[3], 0, 0x7170);
    out2[1] = hc_byte_perm(in[2], 0, 0x7372);
    out2[0] = hc_byte_perm(in[2], 0, 0x7170);
    out1[3] = hc_byte_perm(in[1], 0, 0x7372);
    out1[2] = hc_byte_perm(in[1], 0, 0x7170);
    out1[1] = hc_byte_perm(in[0], 0, 0x7372);
    out1[0] = hc_byte_perm(in[0], 0, 0x7170);
}

__device__ void md4_init_vector(md4_ctx_vector_t* ctx) {
    ctx->h[0] = MD4M_A;
    ctx->h[1] = MD4M_B;
    ctx->h[2] = MD4M_C;
    ctx->h[3] = MD4M_D;

    ctx->w0[0] = 0;
    ctx->w0[1] = 0;
    ctx->w0[2] = 0;
    ctx->w0[3] = 0;
    ctx->w1[0] = 0;
    ctx->w1[1] = 0;
    ctx->w1[2] = 0;
    ctx->w1[3] = 0;
    ctx->w2[0] = 0;
    ctx->w2[1] = 0;
    ctx->w2[2] = 0;
    ctx->w2[3] = 0;
    ctx->w3[0] = 0;
    ctx->w3[1] = 0;
    ctx->w3[2] = 0;
    ctx->w3[3] = 0;

    ctx->len = 0;
}

__device__ void md4_transform_vector(const uint32_t* w0, const uint32_t* w1,
                                     uint32_t* w2, const uint32_t* w3,
                                     uint32_t* digest) {
    uint32_t a = digest[0];
    uint32_t b = digest[1];
    uint32_t c = digest[2];
    uint32_t d = digest[3];

    MD4_STEP(MD4_Fo, a, b, c, d, w0[0], MD4C00, MD4S00);
    MD4_STEP(MD4_Fo, d, a, b, c, w0[1], MD4C00, MD4S01);
    MD4_STEP(MD4_Fo, c, d, a, b, w0[2], MD4C00, MD4S02);
    MD4_STEP(MD4_Fo, b, c, d, a, w0[3], MD4C00, MD4S03);
    MD4_STEP(MD4_Fo, a, b, c, d, w1[0], MD4C00, MD4S00);
    MD4_STEP(MD4_Fo, d, a, b, c, w1[1], MD4C00, MD4S01);
    MD4_STEP(MD4_Fo, c, d, a, b, w1[2], MD4C00, MD4S02);
    MD4_STEP(MD4_Fo, b, c, d, a, w1[3], MD4C00, MD4S03);
    MD4_STEP(MD4_Fo, a, b, c, d, w2[0], MD4C00, MD4S00);
    MD4_STEP(MD4_Fo, d, a, b, c, w2[1], MD4C00, MD4S01);
    MD4_STEP(MD4_Fo, c, d, a, b, w2[2], MD4C00, MD4S02);
    MD4_STEP(MD4_Fo, b, c, d, a, w2[3], MD4C00, MD4S03);
    MD4_STEP(MD4_Fo, a, b, c, d, w3[0], MD4C00, MD4S00);
    MD4_STEP(MD4_Fo, d, a, b, c, w3[1], MD4C00, MD4S01);
    MD4_STEP(MD4_Fo, c, d, a, b, w3[2], MD4C00, MD4S02);
    MD4_STEP(MD4_Fo, b, c, d, a, w3[3], MD4C00, MD4S03);

    MD4_STEP(MD4_Go, a, b, c, d, w0[0], MD4C01, MD4S10);
    MD4_STEP(MD4_Go, d, a, b, c, w1[0], MD4C01, MD4S11);
    MD4_STEP(MD4_Go, c, d, a, b, w2[0], MD4C01, MD4S12);
    MD4_STEP(MD4_Go, b, c, d, a, w3[0], MD4C01, MD4S13);
    MD4_STEP(MD4_Go, a, b, c, d, w0[1], MD4C01, MD4S10);
    MD4_STEP(MD4_Go, d, a, b, c, w1[1], MD4C01, MD4S11);
    MD4_STEP(MD4_Go, c, d, a, b, w2[1], MD4C01, MD4S12);
    MD4_STEP(MD4_Go, b, c, d, a, w3[1], MD4C01, MD4S13);
    MD4_STEP(MD4_Go, a, b, c, d, w0[2], MD4C01, MD4S10);
    MD4_STEP(MD4_Go, d, a, b, c, w1[2], MD4C01, MD4S11);
    MD4_STEP(MD4_Go, c, d, a, b, w2[2], MD4C01, MD4S12);
    MD4_STEP(MD4_Go, b, c, d, a, w3[2], MD4C01, MD4S13);
    MD4_STEP(MD4_Go, a, b, c, d, w0[3], MD4C01, MD4S10);
    MD4_STEP(MD4_Go, d, a, b, c, w1[3], MD4C01, MD4S11);
    MD4_STEP(MD4_Go, c, d, a, b, w2[3], MD4C01, MD4S12);
    MD4_STEP(MD4_Go, b, c, d, a, w3[3], MD4C01, MD4S13);

    MD4_STEP(MD4_H, a, b, c, d, w0[0], MD4C02, MD4S20);
    MD4_STEP(MD4_H, d, a, b, c, w2[0], MD4C02, MD4S21);
    MD4_STEP(MD4_H, c, d, a, b, w1[0], MD4C02, MD4S22);
    MD4_STEP(MD4_H, b, c, d, a, w3[0], MD4C02, MD4S23);
    MD4_STEP(MD4_H, a, b, c, d, w0[2], MD4C02, MD4S20);
    MD4_STEP(MD4_H, d, a, b, c, w2[2], MD4C02, MD4S21);
    MD4_STEP(MD4_H, c, d, a, b, w1[2], MD4C02, MD4S22);
    MD4_STEP(MD4_H, b, c, d, a, w3[2], MD4C02, MD4S23);
    MD4_STEP(MD4_H, a, b, c, d, w0[1], MD4C02, MD4S20);
    MD4_STEP(MD4_H, d, a, b, c, w2[1], MD4C02, MD4S21);
    MD4_STEP(MD4_H, c, d, a, b, w1[1], MD4C02, MD4S22);
    MD4_STEP(MD4_H, b, c, d, a, w3[1], MD4C02, MD4S23);
    MD4_STEP(MD4_H, a, b, c, d, w0[3], MD4C02, MD4S20);
    MD4_STEP(MD4_H, d, a, b, c, w2[3], MD4C02, MD4S21);
    MD4_STEP(MD4_H, c, d, a, b, w1[3], MD4C02, MD4S22);
    MD4_STEP(MD4_H, b, c, d, a, w3[3], MD4C02, MD4S23);

    digest[0] += a;
    digest[1] += b;
    digest[2] += c;
    digest[3] += d;
}

__device__ void md4_update_vector_64(md4_ctx_vector_t* ctx, uint32_t* w0,
                                     uint32_t* w1, uint32_t* w2, uint32_t* w3,
                                     const int len) {
    if (len == 0) return;

    const int pos = ctx->len & 63;

    ctx->len += len;

    if (pos == 0) {
        ctx->w0[0] = w0[0];
        ctx->w0[1] = w0[1];
        ctx->w0[2] = w0[2];
        ctx->w0[3] = w0[3];
        ctx->w1[0] = w1[0];
        ctx->w1[1] = w1[1];
        ctx->w1[2] = w1[2];
        ctx->w1[3] = w1[3];
        ctx->w2[0] = w2[0];
        ctx->w2[1] = w2[1];
        ctx->w2[2] = w2[2];
        ctx->w2[3] = w2[3];
        ctx->w3[0] = w3[0];
        ctx->w3[1] = w3[1];
        ctx->w3[2] = w3[2];
        ctx->w3[3] = w3[3];

        if (len == 64) {
            md4_transform_vector(ctx->w0, ctx->w1, ctx->w2, ctx->w3, ctx->h);

            ctx->w0[0] = 0;
            ctx->w0[1] = 0;
            ctx->w0[2] = 0;
            ctx->w0[3] = 0;
            ctx->w1[0] = 0;
            ctx->w1[1] = 0;
            ctx->w1[2] = 0;
            ctx->w1[3] = 0;
            ctx->w2[0] = 0;
            ctx->w2[1] = 0;
            ctx->w2[2] = 0;
            ctx->w2[3] = 0;
            ctx->w3[0] = 0;
            ctx->w3[1] = 0;
            ctx->w3[2] = 0;
            ctx->w3[3] = 0;
        }
    } else {
        if ((pos + len) < 64) {
            switch_buffer_by_offset_le(w0, w1, w2, w3, pos);

            ctx->w0[0] |= w0[0];
            ctx->w0[1] |= w0[1];
            ctx->w0[2] |= w0[2];
            ctx->w0[3] |= w0[3];
            ctx->w1[0] |= w1[0];
            ctx->w1[1] |= w1[1];
            ctx->w1[2] |= w1[2];
            ctx->w1[3] |= w1[3];
            ctx->w2[0] |= w2[0];
            ctx->w2[1] |= w2[1];
            ctx->w2[2] |= w2[2];
            ctx->w2[3] |= w2[3];
            ctx->w3[0] |= w3[0];
            ctx->w3[1] |= w3[1];
            ctx->w3[2] |= w3[2];
            ctx->w3[3] |= w3[3];
        } else {
            uint32_t c0[4] = {0};
            uint32_t c1[4] = {0};
            uint32_t c2[4] = {0};
            uint32_t c3[4] = {0};

            switch_buffer_by_offset_carry_le(w0, w1, w2, w3, c0, c1, c2, c3,
                                             pos);

            ctx->w0[0] |= w0[0];
            ctx->w0[1] |= w0[1];
            ctx->w0[2] |= w0[2];
            ctx->w0[3] |= w0[3];
            ctx->w1[0] |= w1[0];
            ctx->w1[1] |= w1[1];
            ctx->w1[2] |= w1[2];
            ctx->w1[3] |= w1[3];
            ctx->w2[0] |= w2[0];
            ctx->w2[1] |= w2[1];
            ctx->w2[2] |= w2[2];
            ctx->w2[3] |= w2[3];
            ctx->w3[0] |= w3[0];
            ctx->w3[1] |= w3[1];
            ctx->w3[2] |= w3[2];
            ctx->w3[3] |= w3[3];

            md4_transform_vector(ctx->w0, ctx->w1, ctx->w2, ctx->w3, ctx->h);

            ctx->w0[0] = c0[0];
            ctx->w0[1] = c0[1];
            ctx->w0[2] = c0[2];
            ctx->w0[3] = c0[3];
            ctx->w1[0] = c1[0];
            ctx->w1[1] = c1[1];
            ctx->w1[2] = c1[2];
            ctx->w1[3] = c1[3];
            ctx->w2[0] = c2[0];
            ctx->w2[1] = c2[1];
            ctx->w2[2] = c2[2];
            ctx->w2[3] = c2[3];
            ctx->w3[0] = c3[0];
            ctx->w3[1] = c3[1];
            ctx->w3[2] = c3[2];
            ctx->w3[3] = c3[3];
        }
    }
}

__device__ void md4_update_vector_utf16le(md4_ctx_vector_t* ctx,
                                          const uint32_t* w) {
    uint32_t w0[4];
    uint32_t w1[4];
    uint32_t w2[4];
    uint32_t w3[4];

    int pos1;
    int pos4;

    for (pos1 = 0, pos4 = 0; pos1 < PASSWORD_LENGTH - 32;
         pos1 += 32, pos4 += 8) {
        w0[0] = w[pos4 + 0];
        w0[1] = w[pos4 + 1];
        w0[2] = w[pos4 + 2];
        w0[3] = w[pos4 + 3];
        w1[0] = w[pos4 + 4];
        w1[1] = w[pos4 + 5];
        w1[2] = w[pos4 + 6];
        w1[3] = w[pos4 + 7];

        make_utf16le(w1, w2, w3);
        make_utf16le(w0, w0, w1);

        md4_update_vector_64(ctx, w0, w1, w2, w3, 32 * 2);
    }

    w0[0] = w[pos4 + 0];
    w0[1] = w[pos4 + 1];
    w0[2] = w[pos4 + 2];
    w0[3] = w[pos4 + 3];
    w1[0] = w[pos4 + 4];
    w1[1] = w[pos4 + 5];
    w1[2] = w[pos4 + 6];
    w1[3] = w[pos4 + 7];

    make_utf16le(w1, w2, w3);
    make_utf16le(w0, w0, w1);

    md4_update_vector_64(ctx, w0, w1, w2, w3, (PASSWORD_LENGTH - pos1) * 2);
}

__device__ void append_helper_1x4(uint32_t* r, const uint32_t v,
                                  const uint32_t* m) {
    r[0] |= v & m[0];
    r[1] |= v & m[1];
    r[2] |= v & m[2];
    r[3] |= v & m[3];
}

__device__ void set_mark_1x4(uint32_t* v, const uint32_t offset) {
    const uint32_t c = (offset & 15) / 4;
    const uint32_t r = 0xff << ((offset & 3) * 8);

    v[0] = (c == 0) ? r : 0;
    v[1] = (c == 1) ? r : 0;
    v[2] = (c == 2) ? r : 0;
    v[3] = (c == 3) ? r : 0;
}

__device__ void append_0x80_4x4(uint32_t* w0, uint32_t* w1, uint32_t* w2,
                                uint32_t* w3, const uint32_t offset) {
    uint32_t v[4];

    set_mark_1x4(v, offset);

    const uint32_t offset16 = offset / 16;

    append_helper_1x4(w0, ((offset16 == 0) ? 0x80808080 : 0), v);
    append_helper_1x4(w1, ((offset16 == 1) ? 0x80808080 : 0), v);
    append_helper_1x4(w2, ((offset16 == 2) ? 0x80808080 : 0), v);
    append_helper_1x4(w3, ((offset16 == 3) ? 0x80808080 : 0), v);
}

__device__ void md4_final_vector(md4_ctx_vector_t* ctx) {
    const int pos = ctx->len & 63;

    append_0x80_4x4(ctx->w0, ctx->w1, ctx->w2, ctx->w3, pos);

    if (pos >= 56) {
        md4_transform_vector(ctx->w0, ctx->w1, ctx->w2, ctx->w3, ctx->h);

        ctx->w0[0] = 0;
        ctx->w0[1] = 0;
        ctx->w0[2] = 0;
        ctx->w0[3] = 0;
        ctx->w1[0] = 0;
        ctx->w1[1] = 0;
        ctx->w1[2] = 0;
        ctx->w1[3] = 0;
        ctx->w2[0] = 0;
        ctx->w2[1] = 0;
        ctx->w2[2] = 0;
        ctx->w2[3] = 0;
        ctx->w3[0] = 0;
        ctx->w3[1] = 0;
        ctx->w3[2] = 0;
        ctx->w3[3] = 0;
    }

    ctx->w3[2] = ctx->len * 8;
    ctx->w3[3] = 0;

    md4_transform_vector(ctx->w0, ctx->w1, ctx->w2, ctx->w3, ctx->h);
}

void ntlm(Password* password, Digest* digest) {
    uint32_t w[16] = {0};
    for (uint32_t i = 0, idx = 0; i < PASSWORD_LENGTH; i += 4, idx += 1) {
        w[idx] = password->i[idx];
    }

    md4_ctx_vector_t ctx;
    md4_init_vector(&ctx);
    md4_update_vector_utf16le(&ctx, w);
    md4_final_vector(&ctx);

    digest->i[0] = ctx.h[0];
    digest->i[1] = ctx.h[1];
    digest->i[2] = ctx.h[2];
    digest->i[3] = ctx.h[3];
}

__global__ void ntlm_kernel(Password* passwords, Digest* digests) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    ntlm(&passwords[index], &digests[index]);
}