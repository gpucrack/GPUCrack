#include "chainTest.cuh"

int main(){
    int passwordNumber = getNumberPassword(getTotalSystemMemory());

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    generateChains(passwords, result, passwordNumber, numberOfPass, 10,
                   false, THREAD_PER_BLOCK, true);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
