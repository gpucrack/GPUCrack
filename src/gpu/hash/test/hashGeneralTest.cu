#include "hashGeneralTest.cuh"

int main() {

    int passwordNumber = getNumberPassword(8);

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    hash(passwords, result, passwordNumber, numberOfPass, false);

    //generateNewPasswords(&passwords, passwordNumber);

    //hash(passwords, result, passwordNumber, numberOfPass, false);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return (0);
}