#include "generateTableTest.cuh"

int main(int argc, char *argv[]){
    if (argc != 2) {
        printf("Error: not enough arguments given.\n Usage: 'generateTable mt', where mt is the desired number of end points.");
        exit(1);
    }

    printSignature();

    long domain = pow(CHARSET_LENGTH, sizeof(Password));

    long mtMax = getNumberPassword(atoi(argv[1]));

    long idealM0 = (long)(0.01*(double)domain);

    printf("Ideal m0: %ld\n", idealM0);

    long idealMtMax = (long)((double)idealM0/19.83);

    printf("Ideal mtMax: %ld\n", idealMtMax);

    if (mtMax > idealMtMax) mtMax = idealMtMax;

    printf("mtMax: %ld\n", mtMax);

    long passwordNumber = getM0(mtMax);

    if (passwordNumber > idealM0) printf("m0 is too big\n");

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

    long currentPos = 0;

    for(int i=0; i<numberOfCPUPass; i++) {

        printf("current position: %ld\n",currentPos);

        if (currentPos == 0) createFile((char *) "testStart.bin", true);
        writePoint((char *) "testStart.bin", &passwords, batchSize, t, true, currentPos);

        auto numberOfPass = memoryAnalysisGPU(batchSize);

        generateChains(passwords, batchSize, numberOfPass, t,
                       true, THREAD_PER_BLOCK, false, false);

        printf("Chains generated!\n");

        if (currentPos == 0) createFile((char *) "testEnd.bin", true);
        writePoint((char *) "testEnd.bin", &passwords, batchSize, t, true, currentPos);

        currentPos += batchSize;
    }

    cudaFreeHost(passwords);

    return 0;
}
