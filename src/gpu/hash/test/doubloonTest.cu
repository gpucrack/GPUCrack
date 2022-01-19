#include "doubloonTest.cuh"

int doubloon(int passwordNumber) {

    auto * passwords = generatePasswords(passwordNumber);

    auto * result = parallelized_hash(passwords, passwordNumber);

    free(passwords);

    for(int i=0; i<passwordNumber; i++) {
        for(int j=0; j<passwordNumber; j++) {
            if ((i != j) && (memcmp(result[i].bytes, result[j].bytes, 16) == 0)) {
                printf("TEST FAILED ! COLLISION @ %d, %d\n", i, j);

                for(int n=0; n<HASH_LENGTH; n++){
                    printf("%x",result[i].bytes[n]);
                }
                printf("\n");
                for(int n=0; n<HASH_LENGTH; n++){
                    printf("%x",result[j].bytes[n]);
                }
                printf("\n");

                exit(1);
            }
        }
    }

    free(result);

    printf("TEST PASSED !\n");

    return 0;
}

int main(){
    return doubloon(134217728);
}