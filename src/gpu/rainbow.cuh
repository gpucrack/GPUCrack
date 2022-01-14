#ifndef RAINBOW_CUH
#define RAINBOW_CUH

#define _CRT_SECURE_NO_WARNINGS

#include <assert.h>
#include <math.h>
#include <openssl/sha.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constants.cuh"

/*
    The maximality factor, such as the number of chains at the end of the
    offline phase is alpha*mtmax, where mtmax is the expected maximum number of
    chains in a rainbow table.
*/
#define TABLE_ALPHA 0.952

// The length of a chain in the table.
#define TABLE_T 10000

// The number of tables.
#define TABLE_COUNT 4

/*
    A chain of the rainbow table.
    It contains a startpoint and an endpoint.
*/
typedef struct {
    Password startpoint;
    Password endpoint;
} RainbowChain;

/*
    A rainbow table.
    It's an array of chains, with a length. We store the table number as
    well, since multiple tables need to have different reduction
    functions.
*/
typedef struct {
    RainbowChain* chains;
    unsigned long length;
    unsigned char number;
} RainbowTable;

/*
    A way of comparing two chains to sort them by ascending endpoints.
*/
__device__ int compare_rainbow_chains(const void* p1, const void* p2);

/*
    Helper function to compare passwords.
*/
__device__ inline int pwdcmp(Password* p1, Password* p2);

/*
    Helper function to copy passwords.
*/
__device__ inline void pwdcpy(Password* p1, const Password* p2);

/*
   Returns a char in the [a-zA-Z0-9_-] range given a parameter in the [0-63]
   range. Look at an ASCII table to better understand this function
   (https://www.asciitable.com/).
*/
__device__ char char_in_range(unsigned char n);

/*
    A reduce operation, which returns a plain text for a given `digest`,
    `iteration` and `table_number`.
    The nth `iteration` reduction function should give the nth+1 plain text
    reduction. The `table number` is to discriminate different tables.
    Implementation inspired by https://github.com/jtesta/rainbowcrackalack.
*/
__device__ void reduce_digest(Digest* digest, unsigned long iteration,
                              unsigned char table_number, Password* plain_text);

/*
    Transforms a startpoint from a counter to a valid password.
*/
__device__ void create_startpoint(unsigned long counter, Password* plain_text);

/*
    Deduplicates endpoints in-place, given a sorted rainbow table.
    O(n) complexity.
*/
__device__ void dedup_endpoints(RainbowTable* table);

/*
    Searches the rainbow table for a chain with a specific endpoint using
   binary search, since the endpoints are sorted. O(log n) complexity.
*/
__device__ RainbowChain* binary_search(RainbowTable* table, Password* endpoint);

/*
    Kernel to generate all the chains.
*/
__global__ void ntlm_chain_kernel(RainbowTable* table);

/*
    Pretty prints a rainbow table.
*/
__host__ void print_table(const RainbowTable* table);

#endif