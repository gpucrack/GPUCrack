#include "generateTableTest.cuh"

int main(int argc, char *argv[]){
    if (argc != 2) {
        printf("Error: not enough arguments given.\n Usage: 'generateTable mt', where mt is the desired number of end points.");
        exit(1);
    }

    printSignature();

    int passwordNumber = getM0(getTotalSystemMemory(), atoi(argv[1]));

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    //int t = computeT(getTotalSystemMemory(), atoi(argv[1]));
    int t = 100;

    printf("Number of columns: %d\n\n", t);

    generateChains(passwords, result, passwordNumber, numberOfPass, t,
                   true, THREAD_PER_BLOCK, false);

    printf("Chains generated!\n");

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
