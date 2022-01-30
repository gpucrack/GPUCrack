#include "commons.cuh"

__host__ void generatePasswords(Password ** result, int passwordNumber) {

    cudaError_t status = cudaMallocHost(result, passwordNumber * sizeof(Password), cudaHostAllocDefault);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    generateNewPasswords(result, passwordNumber);
}

__host__ void generateNewPasswords(Password ** result, int passwordNumber) {

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
        for (int i=0; i<PASSWORD_LENGTH; i++) {
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

    size_t freeMem;
    size_t totalMem;
    cudaError_t mem = cudaMemGetInfo(&freeMem, &totalMem);

    // Just to keep a little of memory, just in case
    freeMem -= 500000000;

    // Checking errors on memory detection
    if (mem != cudaSuccess) {
        printf("memory check failed with error \"%s\".\n",
               cudaGetErrorString(mem));
        exit(1);
    }

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
    int batchSize;

    if (numberOfPass > 1)
        batchSize = (passwordNumber / numberOfPass);
        // If we have less than 1 round then the batch size is the number of
        // passwords
    else
        batchSize = passwordNumber;

    return batchSize;
}

__host__ void initEmptyArrays(Password ** passwords, Digest ** results, int passwordNumber) {

    cudaError_t status = cudaMallocHost(passwords, passwordNumber * sizeof(Password), cudaHostAllocDefault);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    status = cudaMallocHost(results, passwordNumber * sizeof(Digest), cudaHostAllocDefault);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

}

__host__ void initArrays(Password ** passwords, Digest ** results, int passwordNumber) {

    generatePasswords(passwords, passwordNumber);

    cudaError_t status = cudaMallocHost(results, passwordNumber * sizeof(Digest), cudaHostAllocDefault);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

}

__device__ __host__ void printDigest(Digest * dig) {

    for(unsigned char byte : dig->bytes){
        printf("%02X", byte);
    }

    printf("\n");
}

__device__ __host__ void printPassword(Password * pwd) {
    for(unsigned char byte : pwd->bytes){
        printf("%c", byte);
    }
    printf("\n");
}

__host__ void createFile(char * name) {
    // Creating file
    std::ofstream file (name);
    printf("CREATING FILE\n");
}

__host__ void writeStarting(char * name, Password ** passwords, int passwordNumber) {
    std::ofstream file;
    file.open(name);

    if(!file.is_open()) {
        printf("ERROR OPENING THE FILE !\n");
    }

    for(int i=0; i<passwordNumber; i++) {
        file << (*passwords)[i].bytes << std::endl;
    }

    printf("FILE WRITTEN\n");

    file.close();
}

__host__ void writeEndingReduction(char * name, Password ** passwords, Digest ** results, int passwordNumber) {
    std::ofstream file;
    file.open(name);

    if(!file.is_open()) {
        printf("ERROR OPENING THE FILE !\n");
    }

    for(int i=0; i<passwordNumber; i++) {
        file << (*passwords)[i].bytes << "-->";
        for(int j=0; j<HASH_LENGTH; j++){
            char buf[HASH_LENGTH];
            sprintf(buf, "%02X", (*results)[i].bytes[j]);
            file << buf;
        }
        file << std::endl;
    }

    printf("FILE WRITTEN\n");
    file.close();
}

__host__ std::ofstream openFile(const char * path) {
    std::ofstream file;
    file.open(path);

    // Check if the file was correctly opened
    if(!file.is_open()) {
        printf("Error: couldn't open file in %s.\n", path);
    }

    return file;
}

__host__ void writeEnding(char * path, Digest ** results, int endpointNumber, bool debug) {
    std::ofstream file = openFile(path);

    // Iterate through every end point
    for(int i=0; i< endpointNumber; i++) {
        // Iterate through every byte of the end point
        for(int j=0; j<HASH_LENGTH; j++){
            char buf[HASH_LENGTH];
            sprintf(buf, "%02X", (*results)[i].bytes[j]); // %02X formats as uppercase hex with leading zeroes
            file << buf;
        }
        file << std::endl;
    }

    if (debug) printf("FILE WRITTEN\n");
    file.close();
}
