#define REFERENCE_SENTENCE1 "1234567"
#include "complianceTest.cuh"

int compliance(int passwordNumber) {

    char REFERENCE_RESULT[32] = {'3','2','8','7','2','7','b','8','1','c','a','0','5','8','0','5'
                               ,'a','6','8','e','f','2','6','a','c','b','2','5','2','0','3','9'};

    auto * passwords = (Password*) malloc(passwordNumber*sizeof(Password));

    // Fill passwords with reference sentence
    for (int j=0; j<passwordNumber; j++) {
        for (int i = 0; i < strlen(REFERENCE_SENTENCE1); i++) {
            passwords[j].bytes[i] = REFERENCE_SENTENCE1[i];
        }
    }

    //Debug
    //printf("\nPASSWORD INSIDE PASSWORDS: ");
    //for (unsigned char i : passwords[0].bytes) {
    //    printf("%c", i);
    //}
    //printf("\n\n");

    auto * result = parallelized_hash(passwords, passwordNumber);

    printf("\n==========COMPLIANCE TEST==========\n");
    printf("RESULTS FROM BASE FUNCTION : \n");
    for (int i=0; i<32; i++) {
        printf("%c", REFERENCE_RESULT[i]);
    }
    printf("\n\n");

    printf("SAMPLE RESULT FROM GPU FUNCTION : \n");
    for(int i=0; i<HASH_LENGTH; i++) {
        printf("%x", result[0].bytes[i]);
    }
    printf("\n\n");

    printf("COMPARING ALL RESULTS TO REFERENCE RESULT:\n");

    for(int i=0; i<passwordNumber; i++){
        for(int j=0; j<HASH_LENGTH-1; j++){
            int comparison = memcmp(REFERENCE_RESULT, result[i].bytes, 16);
            if(comparison != 1) {
                printf("TEST FAILED !\n");
                printf("FAILED @ DIGEST N°%d, CHARACTER N°%d\n", i, j);
                printf("THIS IS THE FAIL SAMPLE: ");
                for(int n=0; n<HASH_LENGTH; n++) {
                    printf("%x", result[i].bytes[n]);
                }
                printf("\n");
                exit(1);
            }
        }
    }

    printf("TEST PASSED !\n");

    free(result);

    printf("====================\n");

    return 0;
}

int main(){
    compliance(268435456);
}