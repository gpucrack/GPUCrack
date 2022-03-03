#include "chainTest.cuh"

int main(){
    int passwordNumber = getNumberPassword(getTotalSystemMemory());

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysisGPU(passwordNumber);

    generateChains(passwords, passwordNumber, numberOfPass, 10,
                   false, THREAD_PER_BLOCK, true, true);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
