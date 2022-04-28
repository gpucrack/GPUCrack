#include "commons.cuh"

__host__ void printSignature() {
    printf("GPUCrack v0.1.3\n"
           "<https://github.com/gpucrack/GPUCrack/>\n\n");
}

__host__ void handleCudaError(cudaError_t status) {
    if (status != cudaSuccess) {
        const char *errorMessage = cudaGetErrorString(status);
        printf("CUDA error: %s.\n", errorMessage);
        exit(1);
    }
}

__host__ void generatePasswords(Password **result, long passwordNumber, unsigned long long offset, unsigned long long tableOffset) {
    handleCudaError(cudaMallocHost(result, passwordNumber * sizeof(Password), cudaHostAllocDefault));
    generateNewPasswords2(result, passwordNumber, offset, tableOffset);
}

__host__ void generateNewPasswords2(Password **result, long passwordNumber, unsigned long long offset, unsigned long long tableOffset) {
    for (long j = offset; j < passwordNumber+offset; j++) {
        // Generate one password
        long counter = j + tableOffset;
        for (unsigned char &byte: (*result)[j-offset].bytes) {
            byte = charset[counter % CHARSET_LENGTH];
            counter /= CHARSET_LENGTH;
        }
    }
}

__host__ void generateNewPasswords(Password **result, int passwordNumber) {

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 61); // define the range

    printf("\nGenerating passwords...\n");
    // Generate all passwords
    for (int j = 0; j < passwordNumber; j++) {
        // Generate one password
        for (unsigned char &byte: (*result)[j].bytes) {
            byte = charset[distr(gen)];
        }
    }
    printf("Done, %d passwords generated\n", passwordNumber);
}

// Returns the number of batch that we need to do
__host__ int memoryAnalysisGPU(long passwordNumber) {

    printf("\n==========GPU MEMORY ANALYSIS==========\n");

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("Number of devices: %d\n", nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1024);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n", (float) (prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n", (float) (prop.sharedMemPerBlock) / 1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n", prop.deviceOverlap ? "yes" : "no");
    }

    // Checking if THREAD_PER_BLOCK is a power of 2 because we will have memory problems otherwise
    if ((ceil(log2(THREAD_PER_BLOCK)) != floor(log2(THREAD_PER_BLOCK)))) {
        printf("Thread per block value is not a power of 2 !\n");
        exit(1);
    }


    // Detect available memory
    size_t freeMem;
    size_t totalMem;
    handleCudaError(cudaMemGetInfo(&freeMem, &totalMem));

    // Just to keep a little of memory, just in case
    freeMem -= 500000000;

    printf("GPU memory available: %ld Megabytes\n", (freeMem / 1000000));

    // Computing memory used by password and result array
    size_t memResult = sizeof(Digest) * passwordNumber;
    size_t memPasswords = sizeof(Password) * passwordNumber;
    size_t memUsed = memPasswords + memResult;

    printf("Memory used by digest array : %ld Megabytes\n", (memResult / 1000000));
    printf("Memory used by password array : %ld Megabytes\n", (memPasswords / 1000000));

    printf("This much memory will be used : %ld Megabytes\n\n", (memUsed / 1000000));

    // We need to determine how many batch we'll do to hash all passwords
    // We need to compute the batch size as well
    auto numberOfPass = (double) ((double) memUsed / (double) freeMem);
    if (numberOfPass < 1) {
        printf("Number of passes : %d\n", 1);
        return 1;
    }

    numberOfPass += 0.5;

    int finalNumberOfPass = (int) numberOfPass;
    if ((finalNumberOfPass % 2) != 0) finalNumberOfPass++;

    printf("Number of passes : %d\n", finalNumberOfPass);

    return finalNumberOfPass;
}

__host__ int memoryAnalysisCPU(long passwordNumber, long passwordMemory) {

    if (passwordNumber > passwordMemory) {
        long numberOfPass = (long) ((long) passwordNumber / (long) passwordMemory);
        return (int) numberOfPass + 1;
    } else {
        return 1;
    }
}

__host__ long computeBatchSize(int numberOfPass, long passwordNumber) {
    // If we have less than 1 round then the batch size is the number of passwords
    if (numberOfPass > 1) return (long)(((long)passwordNumber / (long) numberOfPass)) + 1;
    else return passwordNumber + 1;
}

__host__ void initEmptyArrays(Password **passwords, Digest **results, long passwordNumber) {
    handleCudaError(cudaMallocHost(passwords, passwordNumber * sizeof(Password), cudaHostAllocDefault));
    handleCudaError(cudaMallocHost(results, passwordNumber * sizeof(Digest), cudaHostAllocDefault));
}

__host__ void initArrays(Password **passwords, Digest **results, long passwordNumber) {
    generatePasswords(passwords, passwordNumber, 0, 0);
    handleCudaError(cudaMallocHost(results, passwordNumber * sizeof(Digest), cudaHostAllocDefault));
}

__host__ void initPasswordArray(Password **passwords, long passwordNumber, unsigned long long offset, unsigned long long tableOffset) {
    generatePasswords(passwords, passwordNumber, offset, tableOffset);
}

__device__ __host__ void printDigest(Digest *dig) {
    // Iterate through every byte of the digest
    for (int i = 0; i < HASH_LENGTH; i++) {
        printf("%02X", (*dig).bytes[i]); // %02X formats as uppercase hex with leading zeroes
    }

    //printf("\n");
}

__device__ __host__ void printPassword(Password *pwd) {
    // Iterate through every byte of the password
    for (int i = 0; i < sizeof(Password); i++) {
        printf("%c", (*pwd).bytes[i]);
    }
    //printf("\n");
}

__host__ void createFile(char *path, bool debug) {
    std::ofstream file(path);
    if (debug) printf("\nNew file created: %s.\n", path);
}

__host__ std::ofstream openFile(const char *path) {
    std::ofstream file;
    file.open(path);

    // Check if the file was correctly opened
    if (!file.is_open()) {
        printf("Error: couldn't open file in %s.\n", path);
    }

    return file;
}

__host__ void writePoint(char *path, Password **passwords, long number, int t, int pwd_length, bool debug, long start,
                         unsigned long long totalLength, FILE *file) {

    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    if (start == 0) {

        int numLen = 0;
        long numSave = totalLength;

        while (numSave != 0) {
            numSave /= 10;
            numLen++;
        }

        char num[numLen];
        sprintf(num, "%ld", totalLength);

        int pwdlLen = 1;
        char pwdl[pwdlLen];
        sprintf(pwdl, "%d", pwd_length);

        int tLen = 0;
        int tlSave = t;
        while (tlSave != 0) {
            tlSave /= 10;
            tLen++;
        }
        char tc[tLen];
        sprintf(tc, "%d", t);

        fwrite(&num, sizeof(char) * numLen, 1, file);
        fwrite("\n", sizeof(char), 1, file);
        fwrite(&pwdl, sizeof(char) * pwdlLen, 1, file);
        fwrite("\n", sizeof(char), 1, file);
        fwrite(&tc, sizeof(char) * tLen, 1, file);
        fwrite("\n", sizeof(char), 1, file);
    }

    // Iterate through every point
    for (long i = 0; i < number; i++) {
        fwrite((*passwords)[i].bytes, sizeof(uint8_t) * pwd_length, 1, file);
    }

    program_end = clock();
    program_time_used = ((double) (program_end - program_start)) / CLOCKS_PER_SEC;

    if (debug) printf("File %s was written in %f seconds.\n\n", path, program_time_used);

}

__host__ int computeT(long mtMax, int pwd_length) {
    double domain = pow(CHARSET_LENGTH, pwd_length);

    // Compute t knowing mtMax
    int result = (int) ((double) ((double) (2 * domain) / (double) mtMax) - 2);

    return result;
}

__host__ long getM0(long mtMax, int pwd_length) {
    double mZero;
    long domain = (long) pow(CHARSET_LENGTH, pwd_length);
    printf("domain: %ld\n", domain);

    // Recommended value
    double r = 19.83;

    // mtMax = (double)mt / (double)(1/(double)(1+(double)(1/r)));

    mZero = (double) ((double) r * (double) mtMax);

    printf("m0: %ld\n", (long) mZero);
    return (long) mZero;
}

// Returns the number of line we can store inside goRam
__host__ long getNumberPassword(int goRam, int pwd_length) {
    size_t memLine = pwd_length;

    printf("size of a password: %ld\n", memLine);

    // memUsed = memLine * nbLine -> nbLine = memUsed / memLine
    // totalMem * 1000000000 pour passer de Giga octets à  octets

    long memUsed = (long) ((long) goRam * (long) 1000000000);

    long result = (long) ((long) memUsed / (long) memLine);

    printf("Number of password for %dGo of RAM: %ld\n", goRam, result);
    return result;
}

__host__ int getTotalSystemMemory() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    double value = ((double) (pages * page_size) / 1000000000) - 2;
    if (value > 31.0) return 32;
    else if (value > 15.0) return 16;
    else if (value > 7.0) return 8;
}

__host__ unsigned long long *
computeParameters(unsigned long long *parameters, int argc, char *argv[], bool debug) {

    int pwd_length = atoi(argv[1]);

    long domain = pow(CHARSET_LENGTH, pwd_length);

    long idealM0 = (long)(0.1*(double)domain);

    long idealMtMax = (long)((double)((double)idealM0/(double)19.83));

    long mtMax = getNumberPassword(atoi(argv[3]), pwd_length);

    mtMax = idealMtMax;

    long passwordNumber = idealM0;

    int t = computeT(mtMax, pwd_length);

    auto numberOfCPUPass = memoryAnalysisCPU(passwordNumber, getNumberPassword(getTotalSystemMemory()-9, pwd_length));
    //int numberOfCPUPass = 3;

    long batchSize = computeBatchSize(numberOfCPUPass, passwordNumber);

    if (debug) {
        printf("Number of CPU passes: %d\n", numberOfCPUPass);
        printf("CPU batch size: %ld\n", batchSize);
        printf("Password length: %d\n", pwd_length);
        printf("m0: %ld\n", passwordNumber);
        printf("mtMax: %ld\n", mtMax);
        printf("Number of columns (t): %d\n\n", t);
    }

    parameters[0] = passwordNumber;
    parameters[1] = mtMax;
    parameters[2] = t;
    parameters[3] = numberOfCPUPass;
    parameters[4] = batchSize;

    return parameters;
}

__host__ void generateTables(unsigned long long * parameters, Password * passwords, int argc, char *argv[]) {

    char * path;
    int pwd_length = atoi(argv[1]);
    int tableNumber = atoi(argv[2]);

    // User typed 'generateTable c n mt'
    if (argc == 4) {
        path = (char *) "test";
    }

    // User typed 'generateTable c n mt path'
    if (argc == 5) {
        path = argv[4];
    }

    unsigned long long passwordNumber = parameters[0];
    unsigned long long t = parameters[2];
    unsigned long long numberOfCPUPass = parameters[3];
    unsigned long long batchSize = parameters[4];

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

            generateChains(passwords, batchSize, numberOfPass, t,
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

            cudaFreeHost(passwords);
        }

        fclose(start_file);
        fclose(end_file);

        printf("Engaging filtration...\n");

        // Clean the table by deleting duplicate endpoints
        long *res = filter(startName, endName, startName, endName, numberOfCPUPass, batchSize);
        if (res[2] == res[3]) {
            printf("Filtration done!\n\n");
            printf("The files have been generated with success.\n");
        }
    }
}