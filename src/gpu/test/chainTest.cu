#include "chainTest.cuh"

int main(){
    int passwordNumber = 1073741824;

    printf("%d\n",getTotalSystemMemory());

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    generateChains(passwords, result, passwordNumber, numberOfPass);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
