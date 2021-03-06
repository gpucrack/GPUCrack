#ifndef GPU_CRACK_COMMONS_CUH
#define GPU_CRACK_COMMONS_CUH

#include "constants.cuh"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unistd.h>
#include <cmath>

#include "./hash/hash_functions/ntlm.cuh"
#include "./chains.cuh"
#include "./hash/hash.cuh"
#include "./reduction/reduction.cuh"
#include "../common/filtrationHT.cuh"

extern "C" {
#include "../common/filtration.h"
}

__host__ void handleCudaError(cudaError_t status);

// Generates passwordNumber random passwords, using a 62 character alphanumeric charset.
// The charset contains [a-zA-Z0-9].
__host__ void generateNewPasswords(Password **result, int passwordNumber);

__host__ void generatePasswords(Password **result, long passwordNumber, unsigned long long offset, unsigned long long tableOffset);

__host__ void generateNewPasswords2(Password **result, long passwordNumber, unsigned long long offset, unsigned long long tableOffset);

// Returns the number of batches that we need to do
__host__ int memoryAnalysisGPU(unsigned long long passwordNumber, bool debug);

__host__ int memoryAnalysisCPU(unsigned long long passwordNumber, unsigned long long passwordMemory);

// Returns the size a batch should have
__host__ unsigned long long computeBatchSize(int numberOfPass, unsigned long long passwordNumber);

__host__ void initEmptyArrays(Password **passwords, Digest **results, unsigned long long passwordNumber);

__host__ void initArrays(Password **passwords, Digest **results, unsigned long long passwordNumber);

__host__ void initPasswordArray(Password **passwords, unsigned long long passwordNumber, unsigned long long offset, unsigned long long tableOffset);

/**
 * Prints the name and the version of the product in the console.
 */
__host__ void printSignature();

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
 * @param number the number of points (called m_t if end points, m_0 if start points).
 * @param t the number of columns in a chain.
 * @param pwd_length the length of a password (in characters).
 * @param debug (default: false) to print a message when the file is written.
 */
__host__ void writePoint(char *path, Password **passwords, unsigned long long number, int t, int pwd_length, bool debug, unsigned long long start,
                         unsigned long long totalLength, FILE *file);

/**
 * Function used to compute t
 * @param goRam : How many Go of RAM we will use to compute t
 * @param mtMax : Number of chains we want in the end points (mt < m0)
 * @param pwd_length the length of a password (in characters).
 * @return the number of columns in a chain
 */
__host__ int computeT(unsigned long long mtMax, int pwd_length);

/**
 * Function used to get the maximum number of password to input based on RAM.
 * @param goRam how many Go of RAM we will use to compute t.
 * @param pwd_length the length of a password (in characters).
 * @return maximum m0 based on the RAM (goRAM) available
 */
__host__ long getNumberPassword(int goRam, int pwd_length, bool debug);

/**
 * Function used to get m0 value based on mt and RAM
 * @param goRam : How many Go of RAM we will use to compute t
 * @param mtMax : Number of chains we want in the end points (mt < m0)
 * @param pwd_length the length of a password (in characters).
 * @return m0 based on both memory available (to check if mt is correct) and mt
 */
__host__ long getM0(long mtMax, int pwd_length);

/**
 * Automatically detect system RAM
 * @return system RAM (CPU)
 */
__host__ int getTotalSystemMemory();

__host__ unsigned long long *
computeParameters(unsigned long long *parameters, int argc, char *argv[], bool debug);

__host__ void
generateTables(const unsigned long long *parameters, Password *passwords, int argc, char *argv[], bool debug);

#endif //GPU_CRACK_COMMONS_CUH
