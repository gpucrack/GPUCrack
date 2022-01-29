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

    printf("COMPARING ALL RESULTS TO REFERENCE RESULT\n");

    for (int i = 0; i < 1000; i++) {

        Password * referencePassword = &(passwords[i]);
        auto * referenceDigest = (Digest *)malloc(sizeof(Digest));

        hash(referencePassword, referenceDigest, 1, 1);

        int comparison = memcmp((result[i]).bytes, (*referenceDigest).bytes, HASH_LENGTH);
        if (comparison != 0) {
            printf("TEST FAILED ! %d\n", comparison);
            printf("FAILED @ DIGEST N°%d\n", i);
            printDigest(referenceDigest);
            printf("THIS IS THE FAIL SAMPLE: ");
            printDigest(&(result[i]));
            printf("\n");
            exit(1);
        }

        referencePassword = nullptr;

        free(referencePassword);
        free(referenceDigest);
    }
    printf("TEST PASSED !\n");
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