#include "generalTest.cuh"

int main() {

    int passwordNumber = DEFAULT_PASSWORD_NUMBER;

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    hash(passwords, result, passwordNumber, numberOfPass);

    generateNewPasswords(&passwords, passwordNumber);

    hash(passwords, result, passwordNumber, numberOfPass);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return (0);
}