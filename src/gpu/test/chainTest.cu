#include "chainTest.cuh"

int main(){
    int passwordNumber = getNumberPassword(getTotalSystemMemory());

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    generateChains(passwords, result, passwordNumber, numberOfPass, 1000,
                   false, THREAD_PER_BLOCK, true, true);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
