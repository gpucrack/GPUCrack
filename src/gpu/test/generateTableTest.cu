#include "generateTableTest.cuh"

int main(){
    int passwordNumber = getNumberPassword(8);

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    // int t = computeT(16);

    generateChains(passwords, result, passwordNumber, numberOfPass, 100,
                   true, THREAD_PER_BLOCK);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}