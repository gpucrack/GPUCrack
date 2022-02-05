#include "chainTest.cuh"

int main(){
    int passwordNumber = getNumberPassword(16);

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    // int t = computeT(16);

    generateChains(passwords, result, passwordNumber, numberOfPass, 1000, false);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
