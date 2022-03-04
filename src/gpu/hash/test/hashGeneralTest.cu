#include "hashGeneralTest.cuh"

int main() {

    int passwordNumber = getNumberPassword(6);

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysisGPU(passwordNumber);

    hash(passwords, result, passwordNumber, numberOfPass, false);

    //generateNewPasswords(&passwords, passwordNumber);

    //hash(passwords, result, passwordNumber, numberOfPass, false);

    printPassword(&passwords[0]);
    printf(" --> ");
    printDigest(&result[0]);
    printf("\n");
    printPassword(&passwords[1]);
    printf(" --> ");
    printDigest(&result[1]);
    printf("\n...\n");
    printPassword(&passwords[passwordNumber-2]);
    printf(" --> ");
    printDigest(&result[passwordNumber-2]);
    printf("\n");
    printPassword(&passwords[passwordNumber-1]);
    printf(" --> ");
    printDigest(&result[passwordNumber-1]);
    printf("\n");

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return (0);
}