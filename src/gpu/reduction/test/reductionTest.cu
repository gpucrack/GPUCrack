#include "reductionTest.cuh"

int main() {
    int passwordNumber = getNumberPassword(8);

    Password * passwords;
    Digest * result;

    initEmptyArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    reduce(passwords, result, passwordNumber, numberOfPass, THREAD_PER_BLOCK);

    for(int i=0; i<10; i++) {
        printPassword(&passwords[i]);
        printf("\n");
        reduceDigest(0, &result[i], &passwords[i], 916132832);
        printf("CPU: ");
        printPassword(&passwords[i]);
        printf("\n");
    }

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}