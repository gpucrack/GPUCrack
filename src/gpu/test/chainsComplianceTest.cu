#include "chainsComplianceTest.cuh"


void chainCompliance(int passwordNumber, Password * passwords, Digest * result, int numberOfPass) {

    generateChains(passwords, result, passwordNumber, numberOfPass);

    printf("\n==========SAMPLE OF OUTPUTS==========\n");
    for(int i=passwordNumber-1; i<passwordNumber; i++) {
        printPassword(&(passwords[i]));
        printDigest(&(result[i]));
    }
    printf("\n");

    printf("\n==========COMPLIANCE TEST==========\n");

    Password * referencePassword = &(passwords[passwordNumber-1]);
    Digest * referenceDigest = &(result[passwordNumber-1]);

    printf("RESULTS FROM BASE FUNCTION : \n");
    printPassword(referencePassword);
    printf("\n");
    hash(referencePassword, referenceDigest, 1, 1);
    printf("\nBECOMES:\n");
    printDigest(referenceDigest);
    printf("\n\n");

}

int main() {

    int passwordNumber = DEFAULT_PASSWORD_NUMBER;

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    chainCompliance(passwordNumber, passwords, result, numberOfPass);

    cudaFreeHost(passwords);
    cudaFreeHost(result);
}