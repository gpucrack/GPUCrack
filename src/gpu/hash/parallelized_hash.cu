#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "../constants.cuh"
#include "commons.cuh"

int main() {

    auto numberOfPass = memoryAnalysis();
    int batchSize = computeBatchSize(numberOfPass);

    // Measure global time
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    // Host copies
    Digest * h_results;

    float milliseconds = 0;

    kernel(numberOfPass, batchSize, &milliseconds, &program_start, &h_results);

    printf("HASH RETRIEVED @ %f seconds\n",
           (double)(clock() - program_start) / CLOCKS_PER_SEC);

    // Debug
    printf("SAMPLE OF OUTPUT : ");
    for (unsigned char byte : h_results[666].bytes) {
        printf("%x", byte);
    }
    printf("\n");

    // Compute GPU time and hash rate
    printf("GPU PARALLEL HASH TIME : %f milliseconds\n", milliseconds);
    printf("HASH RATE : %f MH/s\n",
           (PASSWORD_NUMBER / (milliseconds / 1000)) / 1000000);

    // Cleanup
    free(h_results);

    // End and compute total time
    program_end = clock();
    program_time_used =
        ((double)(program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);

    return 0;
}
