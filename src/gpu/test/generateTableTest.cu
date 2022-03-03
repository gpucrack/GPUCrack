#include "generateTableTest.cuh"

int main(int argc, char *argv[]){
    if (argc != 2) {
        printf("Error: not enough arguments given.\n Usage: 'generateTable mt', where mt is the desired number of end points.");
        exit(1);
    }

    printSignature();

    long mtMax = getNumberPassword(atoi(argv[1]));

    printf("mtMax: %ld\n", mtMax);

    long passwordNumber = getM0(mtMax);

    int t = computeT(mtMax);

    printf("Number of columns: %d\n\n", t);

    Password * passwords;

    auto numberOfCPUPass = memoryAnalysisCPU(passwordNumber, getNumberPassword(getTotalSystemMemory()-9));

    printf("Number of CPU passes: %d\n", numberOfCPUPass);

    long batchSize = computeBatchSize(numberOfCPUPass, passwordNumber);

    printf("CPU batch size: %ld\n", batchSize);

    long nbOp = t * passwordNumber;

    printf("Number of crypto op: %ld\n", nbOp);

    initPasswordArray(&passwords, batchSize);

    for(int i=0; i<numberOfCPUPass; i++) {

        auto numberOfPass = memoryAnalysisGPU(batchSize);

        generateChains(passwords, batchSize, numberOfPass, t,
                       true, THREAD_PER_BLOCK, false, false);

        printf("Chains generated!\n");

    }

    cudaFreeHost(passwords);

    return 0;
}
