#include <cstdio>
#include <cstdlib>

#include "../hash.cu"

#define NUMBER_OF_TEST 10
#define MAX_THREAD_NUMBER 1024

void benchmark(int passwordNumber) {
    double maxHashRate;
    int bestValue;

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    //printf("\n==========LAUNCHING BENCHMARK==========\n");

    for (int k = 2; k <= MAX_THREAD_NUMBER; k = k * 2) {
        //printf("\n==========TEST K= %d==========\n", k);
        for (int i = 0; i < NUMBER_OF_TEST; i++) {

            float milliseconds = 0;

            hashTime(passwords, result, passwordNumber, &milliseconds, k,
                     numberOfPass);

            double hashrate = (passwordNumber / (milliseconds / 1000)) / 1000000;

            if (hashrate > maxHashRate) {
                maxHashRate = hashrate;
                bestValue = k;
            }
        }
    }

    printf("MAX HASHRATE : %f\n", maxHashRate);
    printf("BEST THREAD PER BLOCK VALUE : %d\n", bestValue);
    //printf("====================\n");

    cudaFreeHost(passwords);
    cudaFreeHost(result);
}

int main() {

    benchmark(DEFAULT_PASSWORD_NUMBER);

    return 0;
}


