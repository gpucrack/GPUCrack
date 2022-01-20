#include "collisionTest.cuh"

int collision(int passwordNumber) {

    auto * passwords = generatePasswords(passwordNumber);

    auto * result = parallelized_hash(passwords, passwordNumber);

    free(passwords);

    for(int i=0; i<passwordNumber; i++) {
        printf("%d\n", i);
        for(int j=0; j<passwordNumber; j++) {
            if ((i != j) && (memcmp(result[i].bytes, result[j].bytes, 16) == 0)) {
                printf("TEST FAILED ! COLLISION @ %d, %d\n", i, j);

                for(unsigned char byte : result[i].bytes){
                    printf("%x",byte);
                }
                printf("\n");
                for(unsigned char byte : result[j].bytes){
                    printf("%x",byte);
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
    return collision(67108864);
}