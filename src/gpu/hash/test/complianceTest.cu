#include "complianceTest.cuh"

int compliance(int passwordNumber, Password * passwords, Digest * result, int numberOfPass,
               const unsigned char * referencePassword, unsigned char * referenceResult) {

    printf("\n==========FILLING PASSWORD ARRAY WITH REFERENCE PASSWORD==========\n");

    // Fill passwords with reference sentence
    for (int j = 0; j < passwordNumber; j++) {
        for (int i = 0; i < PASSWORD_LENGTH; i++) {
            passwords[j].bytes[i] = referencePassword[i];
        }
    }

    hash(passwords, result, passwordNumber, numberOfPass, false);

    printf("SAMPLE RESULT: \n");
    for (unsigned char byte: passwords[2500].bytes) {
        printf("%c", byte);
    }
    printf("\n====================\n");

    printf("\n==========COMPLIANCE TEST==========\n");
    printf("RESULTS FROM BASE FUNCTION : \n");
    for (int i=0; i<HASH_LENGTH*2; i++) {
        printf("%c", referenceResult[i]);
    }
    printf("\n\n");

    printf("SAMPLE RESULT FROM GPU FUNCTION : \n");
    for (unsigned char byte: result[666].bytes) {
        printf("%x", byte);
    }
    printf("\n\n");

    printf("COMPARING ALL RESULTS TO REFERENCE RESULT\n");

    for (int i = 0; i < passwordNumber; i++) {
        int comparison = memcmp(referenceResult, result[i].bytes, HASH_LENGTH);
        // reference results of comparison
        if (comparison != 0) {
            printf("TEST FAILED ! %d\n", comparison);
            printf("FAILED @ DIGEST N°%d\n", i);
            printf("THIS IS THE FAIL SAMPLE: ");
            for (unsigned char byte: result[i].bytes) {
                printf("%x", byte);
            }
            printf("\n");
            exit(1);
        }
    }

    printf("TEST PASSED !\n");
    printf("====================\n");

    return 0;
}

int main() {

    int passwordNumber = DEFAULT_PASSWORD_NUMBER;

    Password * passwords;
    Digest * result;

    initEmptyArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    unsigned char REFERENCE_PASSWORD2[PASSWORD_LENGTH] = {'a','b','c','d','e','f','g'};

    unsigned char REFERENCE_RESULT2[HASH_LENGTH*2] = {'3','5','2','D','F','E','5','5','1','D','6','2','4',
                                             '5','9','B','2','0','3','4','9','B','7','8','A',
                                             '2','1','A','2','F','3','7'};

    compliance(passwordNumber, passwords, result, numberOfPass, REFERENCE_PASSWORD2, REFERENCE_RESULT2);

    unsigned char REFERENCE_PASSWORD1[PASSWORD_LENGTH] = {'1','2','3','4','5','6','7'};

    unsigned char REFERENCE_RESULT1[HASH_LENGTH*2] = {'3', '2', '8', '7', '2', '7', 'B', '8', '1', 'C', 'A',
                                             '0', '5', '8', '0', '5', 'A','6', '8', 'E', 'F', '2',
                                             '6', 'A', 'C', 'B', '2', '5', '2', '0', '3', '9'};

    compliance(passwordNumber, passwords, result, numberOfPass, REFERENCE_PASSWORD1, REFERENCE_RESULT1);

    cudaFreeHost(result);
    cudaFreeHost(passwords);
}