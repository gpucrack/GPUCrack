#include "generateTableTest.cuh"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Error: not enough arguments given.\n Usage: 'generateTable mt', where mt is the desired number of end points.");
        exit(1);
    }

    printSignature();

    int passwordNumber = getM0(getTotalSystemMemory(), atoi(argv[1]));

    Password *passwords;
    Digest *result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    int t = computeT(getTotalSystemMemory(), atoi(argv[1]));

    printf("Number of columns: %d\n\n", t);

    generateChains(passwords, result, passwordNumber, numberOfPass, t, true, THREAD_PER_BLOCK, false, false);

    printf("Chains generated!\n\n");

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    // Clean the table by deleting duplicate endpoints
    long *res = filter("testStart.bin", "testEnd.bin", "testStart.bin", "testEnd.bin");
    if (res[2] == res[3]) {
        printf("The files have been generated with success.\n");
    }

    return 0;
}
