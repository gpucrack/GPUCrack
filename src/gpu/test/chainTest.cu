#include "chainTest.cuh"

int main(){
    int passwordNumber = DEFAULT_PASSWORD_NUMBER;

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    generateChains(passwords, result, passwordNumber, numberOfPass);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
