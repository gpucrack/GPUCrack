#include "reductionTest.cuh"

int main() {
    int passwordNumber = getNumberPassword(8);

    Password * passwords;
    Digest * result;

    initEmptyArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    reduce(passwords, result, passwordNumber, numberOfPass, THREAD_PER_BLOCK);


    for(int i=0; i<1; i++) {
        printPassword(&passwords[i]);
        printf("\n");
    }

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}