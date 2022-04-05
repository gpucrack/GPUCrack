#include "generateTableTest.cuh"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Error: not enough arguments given.\n"
               "Usage: 'generateTable c n mt (path)', where:\n"
               "     - c is the passwords' length (in characters).\n"
               "     - n is the number of tables to generate.\n"
               "     - mt is the number of end points to be generated.\n"
               "     - (optional) path is the path to the start and end points files to create.\n");
        exit(1);
    }

    printSignature();

    char *path;
    int pwd_length = atoi(argv[1]);

    long domain = pow(CHARSET_LENGTH, pwd_length);

    long idealM0 = (long)(1*(double)domain);

    long idealMtMax = (long)((double)((double)idealM0/(double)19.83));

    long mtMax = getNumberPassword(atoi(argv[3]), pwd_length);

    mtMax = idealMtMax;

    long passwordNumber = idealM0;

    int t = computeT(mtMax, pwd_length);

    int tableNumber = atoi(argv[2]);

    printf("mtMax: %ld\n", mtMax);

    printf("m0: %ld\n", passwordNumber);

    printf("Password length: %d\n", pwd_length);
    printf("Number of columns (t): %d\n\n", t);

    // User typed 'generateTable c n mt'
    if (argc == 4) {
        path = (char *) "test";
    }

    auto numberOfCPUPass = memoryAnalysisCPU(passwordNumber, getNumberPassword(getTotalSystemMemory()-9, pwd_length));

    printf("Number of CPU passes: %d\n", numberOfCPUPass);

    long batchSize = computeBatchSize(numberOfCPUPass, passwordNumber);
    // User typed 'generateTable c n mt path'
    if (argc == 5) {
        path = argv[4];
    }

    printf("CPU batch size: %ld\n", batchSize);

    long nbOp = t * passwordNumber;

    printf("Number of crypto op: %ld\n", nbOp);

    for(int table=0; table < tableNumber; table++) {

        // Generate file name according to table number
        char startName[100];
        char endName[100];
        strcpy(startName, path);
        strcat(startName, "_start_");
        strcpy(endName, path);
        strcat(endName, "_end_");
        char tableChar[10];
        sprintf(tableChar, "%d", table);
        strcat(startName, tableChar);
        strcat(startName, ".bin");
        strcat(endName, tableChar);
        strcat(endName, ".bin");

        // Initialize the passwords array
        Password * passwords;
        initPasswordArray(&passwords, batchSize);

        long currentPos = 0;

        for(int i=0; i<numberOfCPUPass; i++) {

            printf("current position: %ld\n", currentPos);

            if (currentPos == 0) createFile(startName, true);
            writePoint(startName, &passwords, batchSize, t, pwd_length, true, currentPos);

            auto numberOfPass = memoryAnalysisGPU(batchSize);

            generateChains(passwords, batchSize, numberOfPass, t,
                           true, THREAD_PER_BLOCK, false, false, NULL, pwd_length, startName, endName);

            printf("Chains generated!\n");

            if (currentPos == 0) createFile(endName, true);
            writePoint(endName, &passwords, batchSize, t, pwd_length, true, currentPos);

            currentPos += batchSize;
        }
        printf("Engaging filtration...\n");

        // Clean the table by deleting duplicate endpoints
        long *res = filter(startName, endName, startName, endName);
        if (res[2] == res[3]) {
            printf("Filtration done!\n\n");
            printf("The files have been generated with success.\n");
        }

        cudaFreeHost(passwords);
    }

    return 0;
}
