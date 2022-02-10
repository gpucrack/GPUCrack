#include "generateTableTest.cuh"

int main(int argc, char *argv[]){
    if (argc != 1) {
        printf("NOT ENOUGH ARGUMENTS\n");
        exit(1);
    }

    int passwordNumber = getM0(8, atoi(argv[0]));

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    // int t = computeT(16);

    generateChains(passwords, result, passwordNumber, numberOfPass, 100,
                   true, THREAD_PER_BLOCK);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}