#include "reductionTest.cuh"

int main() {
    int passwordNumber = getNumberPassword(1, PASSWORD_LENGTH);

    Password * passwords;
    Digest * result;

    initEmptyArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysisGPU(passwordNumber);

    reduce(passwords, result, passwordNumber, numberOfPass, THREAD_PER_BLOCK, PASSWORD_LENGTH);

    for(int i=0; i<10; i++) {
        printDigest(&result[i]);
        printf(" --> ");
        printPassword(&passwords[i]);
        printf("\n");
        reduceDigest(0, &result[i], &passwords[i], PASSWORD_LENGTH);
        printf("CPU with same column index: ");
        printPassword(&passwords[i]);
        printf("\n");
    }

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}