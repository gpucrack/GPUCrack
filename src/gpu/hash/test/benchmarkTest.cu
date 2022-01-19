#include <cstdio>
#include <cstdlib>

#include "../parallelized_hash.cu"

#define NUMBER_OF_TEST 10
#define MAX_THREAD_NUMBER 1024

int main() {
    double maxHashRate;
    int bestValue;

    Password * passwords = generatePasswords(DEFAULT_PASSWORD_NUMBER);

    printf("\n==========LAUNCHING BENCHMARK==========\n");

    for(int k=2;k<=MAX_THREAD_NUMBER;k=k*2) {
        printf("\n==========TEST K= %d==========\n", k);
        for (int i=0; i<NUMBER_OF_TEST; i++) {

            float milliseconds = 0;

            // Host copies
            auto * result = parallelized_hash_time(passwords, DEFAULT_PASSWORD_NUMBER, &milliseconds);

            free(result);
            double hashrate = (DEFAULT_PASSWORD_NUMBER / (milliseconds / 1000)) / 1000000;

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


