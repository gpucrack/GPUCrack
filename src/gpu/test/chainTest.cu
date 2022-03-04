#include "chainTest.cuh"

int main(){
    int passwordNumber = getNumberPassword(1);

    Password * passwords;
    Digest * result;

    // check
    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysisGPU(passwordNumber);

    hash(passwords, result, passwordNumber, numberOfPass, false);

    printPassword(&passwords[0]);
    printf("\n");
    printDigest(&result[0]);
    printf("\n");

    generateChains(passwords, passwordNumber, numberOfPass, 1,
                   false, THREAD_PER_BLOCK, true, true, result);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
