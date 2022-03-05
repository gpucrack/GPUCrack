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

    int passwordNumber = getM0(getTotalSystemMemory(), atoi(argv[2]), pwd_length);

    Password *passwords;
    Digest *result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    int t = computeT(getTotalSystemMemory(), atoi(argv[2]), pwd_length);

    printf("Password length: %d\n", pwd_length);
    printf("Number of columns (t): %d\n\n", t/2);

    // User typed 'generateTable c mt'
    if(argc == 3) {
        start_path = (char *) "testStart.bin";
        end_path = (char *) "testEnd.bin";
    }

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

    generateChains(passwords, result, passwordNumber, numberOfPass, t, true,
                   THREAD_PER_BLOCK, false, false, pwd_length, start_path, end_path);

    printf("Chains generated!\n\n");

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    // Clean the table by deleting duplicate endpoints
    long *res = filter(start_path, end_path, start_path, end_path);
    if (res[2] == res[3]) {
        printf("The files have been generated with success.\n");
    }

    return 0;
}
