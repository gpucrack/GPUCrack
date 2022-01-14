#define REFERENCE_SENTENCE1 "1234567"
#define REFERENCE_RESULT "328727b81ca0585a68ef26acb252039"

#include "complianceTest.cuh"

int main() {

    auto * passwords = (Password*) malloc(PASSWORD_NUMBER*sizeof(Password));

    // Fill passwords with reference sentence
    for (int j=0; j<PASSWORD_NUMBER; j++) {
        for (int i = 0; i < strlen(REFERENCE_SENTENCE1); i++) {
            passwords[j].bytes[i] = REFERENCE_SENTENCE1[i];
        }
    }

    printf("PASSWORD INSIDE PASSWORDS: ");
    for (unsigned char i : passwords[0].bytes) {
        printf("%c", i);
    }
    printf("\n");

    auto * result = parallelized_hash(passwords);

    printf("RESULTS FROM BASE FUNCTION : ");
    for (unsigned char i : REFERENCE_RESULT) {
        if(i != '\0') printf("%c", i);
    }
    printf("\n");

    printf("SAMPLE RESULT FROM GPU FUNCTION : ");
    for (unsigned char i : result[0].bytes) {
        printf("%x", i);
    }
    printf("\n");

    for(int i=0; i<PASSWORD_NUMBER; i++) {
        for(int j=0;j<HASH_LENGTH;j++){
            if (!strcmp(REFERENCE_RESULT, (char*)result[i].bytes)) {
                printf("TEST FAILED ! @ %d @ %d", i, j);
                exit(1);
            }
        }
    }

    printf("TEST PASSED !");

    return 0;
}