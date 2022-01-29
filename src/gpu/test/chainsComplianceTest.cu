#include "chainsComplianceTest.cuh"


void chainCompliance(int passwordNumber, Password * passwords, Digest * result, int numberOfPass,
                     const unsigned char * referencePassword, unsigned char * referenceResult) {

    printf("\n==========FILLING PASSWORD ARRAY WITH REFERENCE PASSWORD==========\n");

    // Fill passwords with reference sentence
    for (int j = 0; j < passwordNumber; j++) {
        for (int i = 0; i < PASSWORD_LENGTH; i++) {
            passwords[j].bytes[i] = referencePassword[i];
        }
    }

    generateChains(passwords, result, passwordNumber, numberOfPass);

    printf("\n==========OUTPUTS==========\n");
    for(int i=passwordNumber-1; i<passwordNumber; i++) {
        printPassword(&(passwords[i]));
        printDigest(&(result[i]));
    }
    printf("\n");

}

int main() {

    int passwordNumber = DEFAULT_PASSWORD_NUMBER;

    Password * passwords;
    Digest * result;

    initEmptyArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    unsigned char REFERENCE_START_PASSWORD[PASSWORD_LENGTH] = {'1','2','3','4','5','6','7'};

    unsigned char REFERENCE_END_PASSWORD[PASSWORD_LENGTH] = {'1','2','3','4','5','6','7'};

    unsigned char REFERENCE_END_RESULT[HASH_LENGTH*2] = {'3', '2', '8', '7', '2', '7', 'b', '8', '1', 'c', 'a',
                                                      '0', '5', '8', '0', '5', 'a','6', '8', 'e', 'f', '2',
                                                      '6', 'a', 'c', 'b', '2', '5', '2', '0', '3', '9'};
}