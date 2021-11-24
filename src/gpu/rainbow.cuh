#ifndef RAINBOW_H
#define RAINBOW_H

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

// Comment to hide debug prints in release mode.
#define DEBUG_TEST 1

// Handy macro for debug prints.
#if !defined NDEBUG || defined DEBUG_TEST
#define DEBUG_TEST 1
#else
#define DEBUG_TEST 0
#endif
#define DEBUG_PRINT(fmt, ...)                              \
    do {                                                   \
        if (DEBUG_TEST) fprintf(stderr, fmt, __VA_ARGS__); \
    } while (0)

/*
    A chain of the rainbow table.
    It contains a startpoint and an endpoint.
*/
typedef struct {
    Password startpoint;
    Password endpoint;
} RainbowChain;

/*
    A way of comparing two chains to sort them by ascending endpoints.
*/
int compare_rainbow_chains(const void* p1, const void* p2);

/*
    A rainbow table.
    It's an array of chains, with a length. We store the table number as well,
    since multiple tables need to have different reduction functions.
*/
typedef struct {
    RainbowChain* chains;
    unsigned long length;
    unsigned char number;
} RainbowTable;

/*
   Returns a char in the [a-zA-Z0-9_-] range given a parameter in the [0-63]
   range. Look at an ASCII table to better understand this function
   (https://www.asciitable.com/).
*/
char char_in_range(unsigned char n);

/*
    A reduce operation, which returns a plain text for a given `digest`,
    `iteration` and `table_number`.
    The nth `iteration` reduction function should give the nth+1 plain text
    reduction. The `table number` is to discriminate different tables.
    Implementation inspired by https://github.com/jtesta/rainbowcrackalack.
*/
void reduce_digest(unsigned char* digest, unsigned long iteration,
                   unsigned char table_number, char* plain_text);

/*
    Transforms a startpoint from a counter to a valid password.
*/
void create_startpoint(unsigned long counter, char* plain_text);

/*
    Deduplicates endpoints in-place, given a sorted rainbow table.
    O(n) complexity.
*/
void dedup_endpoints(RainbowTable* table);

/*
    Searches the rainbow table for a chain with a specific endpoint using binary
    search, since the endpoints are sorted. O(log n) complexity.
*/
RainbowChain* binary_search(RainbowTable* table, char* endpoint);

/*
    Generates a rainbow table of size `m0*TABLE_T`, where `m0` is the number of
    rows (chains) `TABLE_T` is the number of plain texts in a chain.
    The `table_number` parameter is used to discriminate rainbow tables so
    they're not all similar.
*/
RainbowTable gen_table(unsigned char table_number, unsigned long m0);

/*
    Stores a table to the specified `file_path`.
    No optimizations on the storage are done so the resulting file can be big.
*/
void store_table(RainbowTable* table, const char* file_path);

// Loads a table previously stored on the disk.
RainbowTable load_table(const char* file_path);

// Inserts a chain in the rainbow `table`.
void insert_chain(RainbowTable* table, char* startpoint, char* endpoint);

// Deletes a table.
void del_table(RainbowTable* table);

// Pretty-prints the hash of a digest.
void print_hash(const unsigned char* digest);

// Pretty-prints a rainbow table.
void print_table(const RainbowTable* table);

// Pretty prints the rainbow matrix corresponding to a rainbow table.
void print_matrix(const RainbowTable* table);

/*
    Offline phase of the attack.
    Generates all rainbow tables needed.
*/
void offline(RainbowTable* rainbow_tables);

/*
    Online phase of the attack.
    Uses the pre-generated rainbow tables to guess the plain text of the given
    `digest`.
    Returns in `password` the match if any, or returns an empty string.
*/
void online(RainbowTable* rainbow_tables, unsigned char* digest,
            char* password);

#endif