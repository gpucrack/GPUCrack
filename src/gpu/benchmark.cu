#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "constants.cuh"
#include "commons.cuh"

#define NUMBER_OF_TEST 10
#define MAX_THREAD_NUMBER 1024

int main() {

    auto numberOfPass = memoryAnalysis();

    int batchSize = computeBatchSize(numberOfPass);

    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    printf("LAUNCHING BENCHMARK WITH %d TEST PER THREAD/BLOCK VALUE @ %f seconds\n",NUMBER_OF_TEST,
           (double)(clock() - program_start) / CLOCKS_PER_SEC);

    double maxhashrate;
    int bestValue;

    // Host copies
    Digest * h_results;

    float milliseconds = 0;

    for(int k=8;k<MAX_THREAD_NUMBER;k+=8) {
        for (int i=0; i<NUMBER_OF_TEST; i++) {

            kernel(numberOfPass, batchSize, &milliseconds, &program_start, &h_results);

            // Cleanup
            if (k != (MAX_THREAD_NUMBER - 8)) free(h_results);

            double hashrate = (PASSWORD_NUMBER / (milliseconds / 1000)) / 1000000;

            if(hashrate > maxhashrate){
                maxhashrate = hashrate;
                bestValue = k;
            }
        }
    }

    printf("MAX HASHRATE : %f\n", maxhashrate);
    printf("BEST THREAD PER BLOCK VALUE : %d\n", bestValue);

    printf("SAMPLE OF OUTPUT : ");
    for (unsigned char byte : h_results[666].bytes) {
        printf("%x", byte);
    }
    printf("\n");

    program_end = clock();
    program_time_used =
            ((double)(program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);

    return 0;
}


