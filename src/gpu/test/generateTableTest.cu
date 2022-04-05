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
    int tableNumber = atoi(argv[2]);

    long domain = pow(CHARSET_LENGTH, pwd_length);

    long idealM0 = (long)(0.1*(double)domain);

    long idealMtMax = (long)((double)((double)idealM0/(double)19.83));

    long mtMax = getNumberPassword(atoi(argv[3]), pwd_length);

    mtMax = idealMtMax;

    long passwordNumber = idealM0;

    int t = computeT(mtMax, pwd_length);

    printf("mtMax: %ld\n", mtMax);

    printf("m0: %ld\n", passwordNumber);

    printf("Password length: %d\n", pwd_length);
    printf("Number of columns (t): %d\n\n", t);

    // User typed 'generateTable c n mt'
    if (argc == 4) {
        path = (char *) "test";
    }

    Password * passwords;

    //auto numberOfCPUPass = memoryAnalysisCPU(passwordNumber, getNumberPassword(getTotalSystemMemory()-9, pwd_length));
    int numberOfCPUPass = 3;

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

        unsigned long long tableOffset = table * passwordNumber;

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


        long currentPos = 0;

        FILE * start_file;
        FILE * end_file;

        for(int i=0; i<numberOfCPUPass; i++) {

            initPasswordArray(&passwords, batchSize, currentPos, tableOffset);

            printf("current position: %ld\n", currentPos);

            if (currentPos == 0){
                createFile(startName, true);

                start_file = fopen(startName, "wb");
                if (start_file == nullptr) {
                    printf("Can't open file %s\n", startName);
                    exit(1);
                }
            }

            writePoint(startName, &passwords, batchSize, t, pwd_length, true, currentPos, passwordNumber, start_file);

            auto numberOfPass = memoryAnalysisGPU(batchSize);

            generateChains(passwords+currentPos, batchSize, numberOfPass, t,
                           true, THREAD_PER_BLOCK, false, false, nullptr, pwd_length, startName, endName);

            printf("Chains generated!\n");

            if (currentPos == 0){
                createFile(endName, true);

                end_file = fopen(endName, "wb");
                if (end_file == nullptr) {
                    printf("Can't open file %s\n", endName);
                    exit(1);
                }
            }

            writePoint(endName, &passwords, batchSize, t, pwd_length, true, currentPos, passwordNumber, end_file);

            currentPos += batchSize;
        }

        fclose(start_file);
        fclose(end_file);


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
