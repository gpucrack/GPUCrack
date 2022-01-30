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
__host__ void generateNewPasswords(Password ** result, int passwordNumber);

__host__ void generatePasswords(Password ** result, int passwordNumber);

// Returns the number of batches that we need to do
__host__ int memoryAnalysis(int passwordNumber);

// Returns the size a batch should have
__host__ int computeBatchSize(int numberOfPass, int passwordNumber);

__host__ void initEmptyArrays(Password ** passwords, Digest ** results, int passwordNumber);

__host__ void initArrays(Password ** passwords, Digest ** results, int passwordNumber);

__device__ __host__ void printDigest(Digest * dig);

__device__ __host__ void printPassword(Password * pwd);

__host__ void createFile(char * name);

/**
 * Opens or create a file located at the given path.
 * @param path the path of the file.
 * @return a std::ofstream of the file, which can be used to write data into the file
 */
__host__ std::ofstream openFile(const char * path);

__host__ void writeStarting(char * name, Password ** passwords, int passwordNumber);

__host__ void writeEndingReduction(char * name, Password ** passwords, Digest ** results, int passwordNumber);

/**
 * Writes end points (hashes) into a text file.
 * @param path the path of the file to save.
 * @param results the array containing every end point.
 * @param endpointNumber the number of end points (called m_t).
 */
__host__ void writeEnding(char * path, Digest ** results, int endpointNumber, bool debug = false);

#endif //GPU_CRACK_COMMONS_CUH
