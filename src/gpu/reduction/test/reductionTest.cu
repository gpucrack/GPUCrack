#include "reductionTest.cuh"

int main() {
    int passwordNumber = getNumberPassword(2);

    Password * passwords;
    Digest * result;

    initEmptyArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysisGPU(passwordNumber);

    reduce(passwords, result, passwordNumber, numberOfPass, THREAD_PER_BLOCK);

    for(int i=0; i<10; i++) {
        printDigest(&result[i]);
        printPassword(&passwords[i]);
        printf("\n");
        reduceDigest(0, &result[i], &passwords[i]);
        printf("CPU with different column index: ");
        printPassword(&passwords[i]);
        printf("\n");
    }

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}