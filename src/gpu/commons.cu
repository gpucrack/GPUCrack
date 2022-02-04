#include "commons.cuh"

__host__ void handleCudaError(cudaError_t status) {
    if (status != cudaSuccess) {
        const char *errorMessage = cudaGetErrorString(status);
        printf("CUDA error: %s.\n", errorMessage);
        exit(1);
    }
}

__host__ void generatePasswords(Password **result, int passwordNumber) {
    handleCudaError(cudaMallocHost(result, passwordNumber * sizeof(Password), cudaHostAllocDefault));
    generateNewPasswords(result, passwordNumber);
}

__host__ void generateNewPasswords(Password **result, int passwordNumber) {

    char charSet[62] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                        't', 'u', 'v', 'w', 'x',
                        'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 61); // define the range

    printf("\n==========GENERATING PASSWORDS==========\n");
    // Generate all passwords
    for (int j = 0; j < passwordNumber; j++) {
        // Generate one password
        for (int i = 0; i < PASSWORD_LENGTH; i++) {
            (*result)[j].bytes[i] = charSet[distr(gen)];
        }
    }
    printf("DONE, %d PASSWORDS GENERATED\n", passwordNumber);
}

// Returns the number of batch that we need to do
__host__ int memoryAnalysis(int passwordNumber) {

    printf("\n==========MEMORY ANALYSIS==========\n");

    // Checking if THREAD_PER_BLOCK is a power of 2 because we will have memory problems otherwise
    if ((ceil(log2(THREAD_PER_BLOCK)) != floor(log2(THREAD_PER_BLOCK)))) {
        printf("THREAD PER BLOCK VALUE IS NOT A POWER OF 2 !\n");
        exit(1);
    }


    // Detect available memory
    size_t freeMem;
    size_t totalMem;
    handleCudaError(cudaMemGetInfo(&freeMem, &totalMem));

    // Just to keep a little of memory, just in case
    freeMem -= 500000000;

    printf("MEMORY AVAILABLE : %ld Megabytes\n", (freeMem / 1000000));

    // Computing memory used by password and result array
    size_t memResult = sizeof(Digest) * passwordNumber;
    size_t memPasswords = sizeof(Password) * passwordNumber;
    size_t memUsed = memPasswords + memResult;

    printf("MEMORY USED BY RESULT ARRAY : %ld Megabytes\n",
           (memResult / 1000000));
    printf("MEMORY USED BY PASSWORD ARRAY : %ld Megabytes\n",
           (memPasswords / 1000000));

    printf("THIS MUCH MEMORY WILL BE USED : %ld Megabytes\n",
           (memUsed / 1000000));

    if((memUsed / 1000000000) >= getTotalSystemMemory() - 4) {
        printf("NOT ENOUGH RAM FOR THIS NUMBER OF PASSWORDS !\n");
        exit(1);
    }

    // We need to determine how many batch we'll do to hash all passwords
    // We need to compute the batch size as well
    auto numberOfPass = (double) ((double) memUsed / (double) freeMem);
    if (numberOfPass < 1) return 1;

    numberOfPass += 0.5;

    int finalNumberOfPass = (int) numberOfPass;
    if ((finalNumberOfPass % 2) != 0) finalNumberOfPass++;

    printf("NUMBER OF PASS : %d\n", finalNumberOfPass);

    return finalNumberOfPass;
}

__host__ int computeBatchSize(int numberOfPass, int passwordNumber) {
    // If we have less than 1 round then the batch size is the number of passwords
    if (numberOfPass > 1) return (passwordNumber / numberOfPass);
    else return passwordNumber;
}

__host__ void initEmptyArrays(Password **passwords, Digest **results, int passwordNumber) {
    handleCudaError(cudaMallocHost(passwords, passwordNumber * sizeof(Password), cudaHostAllocDefault));
    handleCudaError(cudaMallocHost(results, passwordNumber * sizeof(Digest), cudaHostAllocDefault));
}

__host__ void initArrays(Password **passwords, Digest **results, int passwordNumber) {
    generatePasswords(passwords, passwordNumber);
    handleCudaError(cudaMallocHost(results, passwordNumber * sizeof(Digest), cudaHostAllocDefault));
}

__device__ __host__ void printDigest(Digest *dig) {
    // Iterate through every byte of the digest
    for (unsigned char byte : dig->bytes) {
        printf("%02X", byte); // %02X formats as uppercase hex with leading zeroes
    }

    printf("\n");
}

__device__ __host__ void printPassword(Password *pwd) {
    // Iterate through every byte of the password
    for (unsigned char byte : pwd->bytes) {
        printf("%c", byte);
    }
    printf("\n");
}

__host__ void createFile(char *path, bool debug) {
    std::ofstream file(path);
    if (debug) printf("New file created: %s.\n", path);
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

__host__ void writeStarting(char *path, Password **passwords, int startNumber, bool debug) {
    std::ofstream file = openFile(path);

    // Iterate through every start point
    for (int i = 0; i < startNumber; i++) {
        file << (*passwords)[i].bytes << std::endl;
    }

    if (debug) printf("The start point file was written.\n");
    file.close();
}


__host__ void writeEndingReduction(char *path, Password **passwords, Digest **results, int endNumber, bool debug) {
    std::ofstream file = openFile(path);

    // Iterate through every end point
    for (int i = 0; i < endNumber; i++) {
        file << (*passwords)[i].bytes << "-->";
        // Iterate through every byte of the end point
        for (int j = 0; j < HASH_LENGTH; j++) {
            char buf[HASH_LENGTH];
            sprintf(buf, "%02X", (*results)[i].bytes[j]); // %02X formats as uppercase hex with leading zeroes
            file << buf;
        }
        file << std::endl;
    }

    if (debug) printf("The end point reduction file was written.\n");
    file.close();
}

__host__ void writeEnding(char *path, Digest **results, int endNumber, bool debug) {
    std::ofstream file = openFile(path);

    // Iterate through every end point
    for (int i = 0; i < endNumber; i++) {
        // Iterate through every byte of the end point
        for (int j = 0; j < HASH_LENGTH; j++) {
            char buf[HASH_LENGTH];
            sprintf(buf, "%02X", (*results)[i].bytes[j]); // %02X formats as uppercase hex with leading zeroes
            file << buf;
        }
        file << std::endl;
    }

    if (debug) printf("The end point file was written.\n");
    file.close();
}

__host__ long computeT(int goRam) {
    int mZero;
    int mtMax;

    // Recommended value
    double r = 19.83;

    // Choosing m0 based on host memory
    if (goRam == 8) mZero = getNumberPassword(8);
    else if (goRam == 12) mZero = getNumberPassword(12);
    else if (goRam == 16) mZero = getNumberPassword(16);
    else if (goRam == 24) mZero = getNumberPassword(24);
    else mZero = getNumberPassword(32);

    // Need to compute mtMax first
    mtMax = mZero / (int) r;

    long domain = 35184372088832;

    // Compute t knowing mtMax
    return ((2*domain) / mtMax) - 2;
}

__host__ int getNumberPassword(int goRam) {
    if (goRam == 8) return 167772160;
    else if (goRam == 12) return 335544320;
    else if (goRam == 16) return 503316480;
    else if (goRam == 24) return 805306368;
    else return 1073741824;
}

__host__ int getTotalSystemMemory() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    double value = ((double)(pages * page_size) / 1000000000) - 2;
    if (value > 31.0) return 32;
    else if (value > 15.0) return 16;
    else if (value > 7.0) return 8;
}