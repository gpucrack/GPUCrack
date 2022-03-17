#include "chainTest.cuh"

int main(int argc, char *argv[]){
    int pwd_length = atoi(argv[1]);

    long domain = pow(CHARSET_LENGTH, pwd_length);

    long idealM0 = (long)(0.2*(double)domain);

    long idealMtMax = (long)((double)idealM0/19.83);

    printf("Ideal m0: %ld\n", idealM0);

    long mtMax = getNumberPassword(atoi(argv[2]), pwd_length);

    printf("Ideal mtMax: %ld\n", idealMtMax);

    if (mtMax > idealMtMax) mtMax = idealMtMax;

    printf("mtMax: %ld\n", mtMax);

    long passwordNumber = getM0(mtMax, pwd_length);

    if (passwordNumber > idealM0) printf("m0 is too big\n");

    int t = computeT(mtMax, pwd_length);

    printf("Password length: %d\n", pwd_length);
    printf("Number of columns (t): %d\n\n", t);

    Password * passwords;
    Digest * result;

    // check
    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysisGPU(passwordNumber);

    char * start_path = (char *) "testStart.bin";
    char * end_path = (char *) "testEnd.bin";

    passwords[0].bytes[0] = 'Z';
    passwords[0].bytes[1] = 'm';
    passwords[0].bytes[2] = 'd';

    // Adjust t depending on the chain length you want to test
    generateChains(passwords, passwordNumber, numberOfPass, t,
                   false, THREAD_PER_BLOCK, true, true, result, PASSWORD_LENGTH, start_path, end_path);

    printf("Should be first password inside endpoints:\n");
    printPassword(&passwords[0]);
    printf("\n");


    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
