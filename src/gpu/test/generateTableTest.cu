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

    char *start_path;
    char *end_path;
    int pwd_length = atoi(argv[1]);

    long domain = pow(CHARSET_LENGTH, pwd_length);

    long idealM0 = (long)(0.1*(double)domain);

    long idealMtMax = (long)((double)((double)idealM0/(double)19.83));

    long mtMax = getNumberPassword(atoi(argv[2]), pwd_length);

    mtMax = idealMtMax;

    long passwordNumber = idealM0;

    int t = computeT(mtMax, pwd_length);

    printf("mtMax: %ld\n", mtMax);

    printf("m0: %ld\n", passwordNumber);

    printf("Password length: %d\n", pwd_length);
    printf("Number of columns (t): %d\n\n", t);

    // User typed 'generateTable c mt'
    if (argc == 3) {
        start_path = (char *) "testStart.bin";
        end_path = (char *) "testEnd.bin";
    }

    Password * passwords;

    //auto numberOfCPUPass = memoryAnalysisCPU(passwordNumber, getNumberPassword(getTotalSystemMemory()-9, pwd_length));
    int numberOfCPUPass = 3;

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

    printf("CPU batch size: %ld\n", batchSize);

    long nbOp = t * passwordNumber;

    printf("Number of crypto op: %ld\n", nbOp);

    long currentPos = 0;

    FILE * start_file;
    FILE * end_file;

    for(int i=0; i<numberOfCPUPass; i++) {

        initPasswordArray(&passwords, batchSize, currentPos);

        printf("current position: %ld\n", currentPos);

        if (currentPos == 0){
            createFile(start_path, true);

            start_file = fopen(start_path, "wb");
            if (start_file == nullptr) {
                printf("Can't open file %s\n", start_path);
                exit(1);
            }
        }

        writePoint(start_path, &passwords, batchSize, t, pwd_length, true, currentPos, passwordNumber, start_file);

        auto numberOfPass = memoryAnalysisGPU(batchSize);

        generateChains(passwords+currentPos, batchSize, numberOfPass, t,
                       true, THREAD_PER_BLOCK, false, false, nullptr, pwd_length, start_path, end_path);

        printf("Chains generated!\n");

        if (currentPos == 0){
            createFile(end_path, true);

            end_file = fopen(end_path, "wb");
            if (end_file == nullptr) {
                printf("Can't open file %s\n", end_path);
                exit(1);
            }
        }

        writePoint(end_path, &passwords, batchSize, t, pwd_length, true, currentPos, passwordNumber, end_file);

        currentPos += batchSize;
    }

    fclose(start_file);
    fclose(end_file);


    printf("Engaging filtration...\n");

    /*
    // Clean the table by deleting duplicate endpoints
    long *res = filter(start_path, end_path, start_path, end_path, 0, batchSize/2);
    if (res[2] == res[3]) {
        printf("Filtration done!\n\n");
        printf("The files have been generated with success.\n");
    }
     */


    cudaFreeHost(passwords);

    return 0;
}
