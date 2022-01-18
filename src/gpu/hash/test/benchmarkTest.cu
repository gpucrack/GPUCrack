#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "../parallelized_hash.cu"
#include "../../constants.cuh"
#include "../commons.cuh"

#define NUMBER_OF_TEST 10
#define MAX_THREAD_NUMBER 2048

int main() {
    double maxHashRate;
    int bestValue;

    Password * passwords = generatePasswords(DEFAULTPASSWORDNUMBER);

    printf("\n==========LAUNCHING BENCHMARK==========\n");

    for(int k=2;k<MAX_THREAD_NUMBER;k=k*2) {
        printf("\n==========TEST K= %d==========\n", k);
        for (int i=0; i<NUMBER_OF_TEST; i++) {

            float milliseconds = 0;

            // Host copies
            auto * result = parallelized_hash(passwords, DEFAULTPASSWORDNUMBER);

            free(result);
            double hashrate = (DEFAULTPASSWORDNUMBER / (milliseconds / 1000)) / 1000000;

            if(hashrate > maxHashRate){
                maxHashRate = hashrate;
                bestValue = k;
            }
        }
    }

    printf("MAX HASHRATE : %f\n", maxHashRate);
    printf("BEST THREAD PER BLOCK VALUE : %d\n", bestValue);
    printf("====================\n");


    return 0;
}


