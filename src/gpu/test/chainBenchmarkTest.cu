#include "chainBenchmarkTest.cuh"

int main() {
    int passwordNumber = getNumberPassword(getTotalSystemMemory());

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    int numberOfColumn = 1000;

    generateChains(passwords, result, passwordNumber,
                   numberOfPass, numberOfColumn, false, THREAD_PER_BLOCK, true, false);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}