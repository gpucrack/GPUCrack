#include "hash.cuh"

void hash(Password * h_passwords, Digest * h_results, int passwordNumber, int numberOfPass) {

    int batchSize = computeBatchSize(numberOfPass, passwordNumber);

    // Measure global time
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    float milliseconds = 0;

    hashKernel(numberOfPass, batchSize, &milliseconds, &program_start, &h_results, &h_passwords, passwordNumber,
               THREAD_PER_BLOCK);

    // Compute GPU time and hash rate
    printf("GPU PARALLEL HASH TIME : %f milliseconds\n", milliseconds);
    printf("HASH RATE : %f MH/s\n",
           (passwordNumber / (milliseconds / 1000)) / 1000000);

    // End and compute total time
    program_end = clock();
    program_time_used =
            ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);
}


// Another version using a time variable, so we can retrieve its value
void hashTime(Password *h_passwords, Digest * h_results, int passwordNumber, float *milliseconds,
              int threadPerBlock, int numberOfPass) {

    int batchSize = computeBatchSize(numberOfPass, passwordNumber);

    // Measure global time
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    hashKernel(numberOfPass, batchSize, milliseconds, &program_start, &h_results, &h_passwords, passwordNumber,
               threadPerBlock);

    // Compute GPU time and hash rate
    printf("GPU PARALLEL HASH TIME : %f milliseconds\n", *milliseconds);
    printf("HASH RATE : %f MH/s\n",
           (passwordNumber / (*milliseconds / 1000)) / 1000000);

    // End and compute total time
    program_end = clock();
    program_time_used =
            ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);
}
