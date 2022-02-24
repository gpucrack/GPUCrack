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

    int t = computeT(getTotalSystemMemory(), atoi(argv[1]));

    printf("Number of columns: %d\n\n", t);

    generateChains(passwords, result, passwordNumber, numberOfPass, t,
                   true, THREAD_PER_BLOCK, false);

    printf("Chains generated!\n");


    printPassword(&passwords[0]);
    printf(" --> ");
    printDigest(&result[0]);
    printf(" --> ");
    printPassword(&passwords[1]);
    printf("...\n");
    printDigest(&result[passwordNumber-1]);
    printf(" --> ");
    printPassword(&passwords[passwordNumber-1]);
    printf("\n");

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
