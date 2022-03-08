#include "generateTableTest.cuh"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Error: not enough arguments given.\n"
               "Usage: 'generateTable c mt (startpath) (endpath)', where:\n"
               "     - c is the passwords' length (in characters).\n"
               "     - mt is the number of end points to be generated.\n"
               "     - (optional) startpath is the path to the start points file to create.\n"
               "     - (optional) endpath is the path to the end points file to create.\n");
        exit(1);
    }

    printSignature();

    long domain = pow(CHARSET_LENGTH, sizeof(Password));

    long mtMax = getNumberPassword(atoi(argv[1]));

    long idealM0 = (long)(0.01*(double)domain);

    printf("Ideal m0: %ld\n", idealM0);

    long idealMtMax = (long)((double)idealM0/19.83);
    char *start_path;
    char *end_path;
    int pwd_length = atoi(argv[1]);

    int passwordNumber = getM0(getTotalSystemMemory(), atoi(argv[2]), pwd_length);

    printf("Ideal mtMax: %ld\n", idealMtMax);

    if (mtMax > idealMtMax) mtMax = idealMtMax;

    printf("mtMax: %ld\n", mtMax);

    long passwordNumber = getM0(mtMax);

    if (passwordNumber > idealM0) printf("m0 is too big\n");

    int t = computeT(mtMax);
    int t = computeT(getTotalSystemMemory(), atoi(argv[2]), pwd_length);

    printf("Password length: %d\n", pwd_length);
    printf("Number of columns (t): %d\n\n", t / 2);

    // User typed 'generateTable c mt'
    if (argc == 3) {
        start_path = (char *) "testStart.bin";
        end_path = (char *) "testEnd.bin";
    }

    Password * passwords;

    auto numberOfCPUPass = memoryAnalysisCPU(passwordNumber, getNumberPassword(getTotalSystemMemory()-9));

    printf("Number of CPU passes: %d\n", numberOfCPUPass);

    long batchSize = computeBatchSize(numberOfCPUPass, passwordNumber);
    // User typed 'generateTable c mt startpath'
    if (argc == 4) {
        start_path = argv[3];
        end_path = (char *) "testEnd.bin";
    }

    // User typed 'generateTable c mt startpath endpath'
    if (argc == 5) {
        start_path = argv[3];
        end_path = argv[4];
    }

    generateChains(passwords, result, passwordNumber, numberOfPass, t, true, THREAD_PER_BLOCK, false, false, pwd_length,
                   start_path, end_path);

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
                       true, THREAD_PER_BLOCK, false, false, NULL);

        printf("Chains generated!\n");

        if (currentPos == 0) createFile((char *) "testEnd.bin", true);
        writePoint((char *) "testEnd.bin", &passwords, batchSize, t, true, currentPos);

        /*
        // Clean the table by deleting duplicate endpoints
        long *res = filter("testStart.bin", "testEnd.bin", "testStart.bin", "testEnd.bin");
        if (res[2] == res[3]) {
            printf("The files have been generated with success.\n");
        }
         */

        currentPos += batchSize;
    printf("Engaging filtration...\n");

    // Clean the table by deleting duplicate endpoints
    long *res = filter(start_path, end_path, start_path, end_path);
    if (res[2] == res[3]) {
        printf("Filtration done!\n\n");
        printf("The files have been generated with success.\n");
    }

    cudaFreeHost(passwords);

    return 0;
}
