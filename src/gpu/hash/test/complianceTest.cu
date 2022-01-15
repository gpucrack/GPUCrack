#define REFERENCE_SENTENCE1 "1234567"
#define REFERENCE_RESULT "328727b81Ca05805a68ef26acb252039"
#include "complianceTest.cuh"

int compliance(int passwordNumber, int passwordPerKernel) {

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

    auto * result = parallelized_hash(passwords, passwordNumber, passwordPerKernel);

    printf("\n==========COMPLIANCE TEST==========\n");
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

    printf("COMPARING ALL RESULTS TO REFERENCE RESULT\n");

    for(int i=0; i<passwordNumber; i++) {
        int comparison = strcmp((const char *)result[i].bytes, REFERENCE_RESULT);
        if((comparison) != 0){
            printf("TEST FAILED %d !\n", comparison);
            exit(1);
        }
    }

    printf("TEST PASSED !");
    printf("\n");

    free(result);

    printf("====================\n");

    return 0;
}

int main(){
    compliance(536870912, 2);
}