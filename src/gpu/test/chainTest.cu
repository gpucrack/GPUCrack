#include "chainTest.cuh"

int main(){
    int passwordNumber = getNumberPassword(1);

    Password * passwords;
    Digest * result;

    // check
    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysisGPU(passwordNumber);

    // Adjust t depending on the chain length you want to test
    generateChains(passwords, passwordNumber, numberOfPass, 3964,
                   false, THREAD_PER_BLOCK, true, true, result);

    printf("Should be first password inside endpoints:\n");
    printPassword(&passwords[0]);
    printf("\n");


    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
