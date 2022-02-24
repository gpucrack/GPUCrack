#include "commons.cuh"

__host__ void printSignature() {
    printf("GPUCrack v0.1.1\n"
           "<https://github.com/gpucrack/GPUCrack/>\n\n");
}

__host__ void handleCudaError(cudaError_t status) {
    if (status != cudaSuccess) {
        const char *errorMessage = cudaGetErrorString(status);
        printf("CUDA error: %s.\n", errorMessage);
        exit(1);
    }
}

__host__ void generatePasswords(Password **result, int passwordNumber) {
    handleCudaError(cudaMallocHost(result, passwordNumber * sizeof(Password), cudaHostAllocDefault));
    generateNewPasswords2(result, passwordNumber);
}

__host__ void generateNewPasswords2(Password **result, int passwordNumber) {
    char charset[62] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                        't', 'u', 'v', 'w', 'x',
                        'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
    char charsetLength = 61;

    for (int j = 0; j < passwordNumber; j++) {
        // Generate one password
        int counter = j;
        for (unsigned char & byte : (*result)[j].bytes) {
            byte = charset[ counter % charsetLength];
            counter /= charsetLength;
        }
    }
}

__host__ void generateNewPasswords(Password **result, int passwordNumber) {

    char charSet[62] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                        't', 'u', 'v', 'w', 'x',
                        'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 61); // define the range

    printf("\nGenerating passwords...\n");
    // Generate all passwords
    for (int j = 0; j < passwordNumber; j++) {
        // Generate one password
        for (unsigned char & byte : (*result)[j].bytes) {
            byte = charSet[distr(gen)];
        }
    }
    printf("Done, %d passwords generated\n", passwordNumber);
}

// Returns the number of batch that we need to do
__host__ int memoryAnalysis(int passwordNumber) {

    printf("\n==========GPU MEMORY ANALYSIS==========\n");

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("Number of devices: %d\n", nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
               prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
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

    printf("Memory used by digest array : %ld Megabytes\n",
           (memResult / 1000000));
    printf("Memory used by password array : %ld Megabytes\n",
           (memPasswords / 1000000));

    printf("This much memory will be used : %ld Megabytes\n\n",
           (memUsed / 1000000));

    if((memUsed / 1000000000) >= getTotalSystemMemory() - 4) {
        printf("Not enough GPU memory for this number of passwords !\n");
        exit(1);
    }

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

    //printf("\n");
}

__device__ __host__ void printPassword(Password *pwd) {
    // Iterate through every byte of the password
    for (unsigned char byte : pwd->bytes) {
        printf("%c", byte);
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

__host__ void writePoint(char *path, Password **passwords, int number, int t, bool debug) {
    std::ofstream file = openFile(path);

    file << number << std::endl;
    file << PASSWORD_LENGTH << std::endl;
    file << t << std::endl;

    // Iterate through every point
    for (int i = 0; i < number; i++) {
        for(int j=0; j < PASSWORD_LENGTH; j++) {
            file << (*passwords)[i].bytes[j];
        }
        file << std::endl;
    }

    if (debug) printf("The point file was written.\n\n");
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

__host__ int computeT(int goRam, int mt) {
    double mtMax;

    // Recommended value
    double r = 19.83;

    // Need to compute mtMax first
    mtMax = (double)mt / (double)(1/(double)(1+(double)(1/r)));

    double domain = pow(62, PASSWORD_LENGTH);

    // Compute t knowing mtMax
    int result = (int)((2*domain) / (int)mtMax) - 2;
    if (result < 1) return 1;
    else return result;
}

__host__ int getM0(int goRam, int mt) {
    double mZero;
    double mtMax;
    double tmpMZero;

    // Recommended value
    double r = 19.83;

    // Choosing m0 based on host memory
    mZero = getNumberPassword(goRam);

    // Need to compute mtMax first
    mtMax = (double)mt / (double)(1/(double)(1+(double)(1/r)));

    tmpMZero = r * mtMax;

    if (tmpMZero > mZero && pow(62, PASSWORD_LENGTH) > tmpMZero) {
        printf("Chosen mt require a bigger m0 than memory available!\n");
        exit(1);
    }else {
        printf("m0: %d\n", (int)mZero);
        return (int)mZero;
    }
}

__host__ int getNumberPassword(int goRam) {

    double domain = pow(62, PASSWORD_LENGTH);

    size_t memLine = sizeof(Password) + sizeof(Digest);

    // memUsed = memLine * nbLine -> nbLine = memUsed / memLine
    // memUsed = totalMem - 4 Go
    // totalMem * 1000000000 pour passer de Giga octets à  octets

    long memUsed = ((long)goRam * (long)1000000000) - ((long)2 * (long)1000000000);

    int result = (int)pow(2, (int)log2((int)((long)memUsed / (int)memLine)));

    if (result > domain){
        result = (int)pow(2, ceil((int)log((int)domain)/log(2)));
    }

    printf("M0: %d\n", result);
    return result;
}

__host__ int getTotalSystemMemory() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    double value = ((double)(pages * page_size) / 1000000000) - 2;
    if (value > 31.0) return 32;
    else if (value > 15.0) return 16;
    else if (value > 7.0) return 8;
}