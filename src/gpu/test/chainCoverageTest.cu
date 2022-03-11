#include "chainCoverageTest.cuh"

int main(){

    int pwd_length = 3;

    getNumberPassword(1, pwd_length);

    long domain = pow(CHARSET_LENGTH, pwd_length);

    printf("Domain: %ld\n", domain);

    long passwordNumber = (long)(0.01*(double)domain);

    printf("m0: %ld\n", passwordNumber);

    long mtMax = (long)((double)passwordNumber/19.83);

    Password * passwords;

    int t = computeT(mtMax, pwd_length);

    printf("Password to be stored in ram: %ld\n", passwordNumber*t);

    printf("Number of columns (t): %d\n\n", t);

    printf("mtMax: %ld\n", mtMax);

    initPasswordArray(&passwords, passwordNumber*t);

    char * start_path = (char *) "testStart.bin";
    char * end_path = (char *) "testEnd.bin";

    // Adjust t depending on the chain length you want to test

    for(int i=0; i<t;i++) {
        printf("%d: ", i);
        generateChains(&(passwords[i*passwordNumber]), passwordNumber, 1, i+1,
                       false, THREAD_PER_BLOCK, false, false, NULL, pwd_length, start_path, end_path);
        printPassword(&(passwords[i*passwordNumber]));
        printf("\n");
    }

    printf("Generation done\n");

    char charset[62] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                        't', 'u', 'v', 'w', 'x',
                        'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
    char charsetLength = 61;

    int nbFound = 0;
    Password * result = (Password *)malloc(pwd_length);

    for(int i=0; i<domain; i++){
        // Generate one password
        long counter = i;
        for (unsigned char & byte : (*result).bytes) {
            byte = charset[ counter % charsetLength];
            counter /= charsetLength;
        }

        for(int k=0; k< passwordNumber*t; k++){
            if(memcmp(&(*result).bytes, &(passwords[k].bytes), pwd_length) == 0){
                nbFound++;
                break;
            }
        }
        if ((i % 1000) == 0) printf("%d \n", i);
    }
    printPassword(&passwords[0]);
    printf("\n");
    printf("Number of passwords found: %d\n", nbFound);
    printf("Coverage: %f\n", ((double)(double)nbFound / (double)domain) * 100);

    cudaFreeHost(passwords);

    return 0;
}
