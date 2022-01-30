#ifndef GPU_CRACK_COMMONS_CUH
#define GPU_CRACK_COMMONS_CUH

#include "constants.cuh"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "./hash/hash_functions/ntlm.cuh"
#include "./chains.cuh"
#include "./hash/hash.cuh"

// Generates passwordNumber random passwords, using a 62 character alphanumeric charset.
// The charset contains [a-zA-Z0-9].
__host__ void generateNewPasswords(Password **result, int passwordNumber);

__host__ void generatePasswords(Password **result, int passwordNumber);

// Returns the number of batches that we need to do
__host__ int memoryAnalysis(int passwordNumber);

// Returns the size a batch should have
__host__ int computeBatchSize(int numberOfPass, int passwordNumber);

__host__ void initEmptyArrays(Password **passwords, Digest **results, int passwordNumber);

__host__ void initArrays(Password **passwords, Digest **results, int passwordNumber);

/**
 * Prints a single digest in the console.
 * @param dig the digest to display.
 */
__device__ __host__ void printDigest(Digest *dig);

/**
 * Prints a single password in the console.
 * @param pwd the password to display.
 */
__device__ __host__ void printPassword(Password *pwd);

/**
 * Creates a file at the given path.
 * @param path the path of the file to be created.
 * @param debug (default: false) to print a message when the file is created.
 */
__host__ void createFile(char *path, bool debug = false);

/**
 * Opens the file located at the given path.
 * @param path the path of the file.
 * @return a std::ofstream of the file, which can be used to write data into the file
 */
__host__ std::ofstream openFile(const char *path);

/**
 * Writes start points (passwords) into a text file.
 * @param path the path of the file to save.
 * @param passwords the array containing every password located in first row.
 * @param startNumber the number of start points (called m_0).
 * @param debug (default: false) to print a message when the file is written.
 */
__host__ void writeStarting(char *path, Password **passwords, int startNumber, bool debug = false);

/**
 * Writes the last reductions of a table (password --> end point) into a text file.
 * @param path the path of the file to save.
 * @param passwords the array containing every password located in row t-1.
 * @param results the array containing every end point.
 * @param endNumber the number of end points (called m_t).
 * @param debug (default: false) to print a message when the file is written.
 */
__host__ void
writeEndingReduction(char *path, Password **passwords, Digest **results, int endNumber, bool debug = false);

/**
 * Writes end points (hashes) into a text file.
 * @param path the path of the file to save.
 * @param results the array containing every end point.
 * @param endNumber the number of end points (called m_t).
 * @param debug (default: false) to print a message when the file is written.
 */
__host__ void writeEnding(char *path, Digest **results, int endNumber, bool debug = false);

#endif //GPU_CRACK_COMMONS_CUH
