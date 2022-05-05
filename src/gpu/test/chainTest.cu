#include "chainTest.cuh"

int main(int argc, char *argv[]){
    char *start_path;
    char *end_path;
    int pwd_length = atoi(argv[1]);

    long domain = pow(CHARSET_LENGTH, pwd_length);

    long idealM0 = (long)(0.001*(double)domain);

    long idealMtMax = (long)((double)((double)idealM0/(double)19.83));

    long mtMax = getNumberPassword(atoi(argv[2]), pwd_length, false);

    mtMax = idealMtMax;

    long passwordNumber = idealM0;

    int t = computeT(mtMax, pwd_length);

    printf("mtMax: %ld\n", mtMax);

    printf("m0: %ld\n", passwordNumber);

    printf("Password length: %d\n", pwd_length);
    printf("Number of columns (t): %d\n\n", t);

    Password * passwords;
    Digest * result;

    // check
    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysisGPU(passwordNumber, false);

    // Adjust t depending on the chain length you want to test
    generateChains(passwords, passwordNumber, numberOfPass, t,
                   false, THREAD_PER_BLOCK, true, true, result, PASSWORD_LENGTH, start_path, end_path, nullptr, 0);

    printf("Should be first password inside endpoints:\n");
    printPassword(&passwords[0]);
    printf("\n");


    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
