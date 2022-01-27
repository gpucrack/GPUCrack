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

__host__ void hashKernel(const int numberOfPass, int batchSize,
                         float *milliseconds, const clock_t *program_start,
                         Digest **h_results, Password **h_passwords, int passwordNumber,
                         int threadPerBlock) {

    // Device copies
    Digest *d_results;
    Password *d_passwords;

    // Measure GPU time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaStream_t stream1;

    cudaStreamCreate(&stream1);

    int passwordRemaining = passwordNumber;
    int currentIndex = 0;

    // Main loop, we add +1 to be sure to do all the batches in case
    // we have 2.5 for example, it'll be 3 passes
    for (int i = 0; i<numberOfPass; i++) {
        // Temporary variable to measure GPU time inside this loop
        float tempMilli = 0;

        // If the currentIndex to save result is greater than the number of
        // password we must stop
        if (currentIndex >= passwordNumber) break;

        // If we have less than batchSize password to hash, then hash them all
        // but modify the batchSize to avoid index errors
        if (passwordRemaining <= batchSize) batchSize = passwordRemaining;

        // GPU Malloc for the password array, size is batchSize
        cudaMalloc(&d_passwords, sizeof(Password) * batchSize);
        cudaMalloc(&d_results, sizeof(Digest) * batchSize);

        Password *source = *h_passwords;

        // Device copies
        cudaMemcpyAsync(d_passwords, &(source[currentIndex]), sizeof(Password) * batchSize,
                   cudaMemcpyHostToDevice, stream1);

        cudaEventRecord(start);
        ntlm_kernel<<<((batchSize) / threadPerBlock), threadPerBlock, 0, stream1>>>(
                d_passwords, d_results);
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        // Necessary procedure to record time and store the elasped time in
        // tempMilli
        cudaEventElapsedTime(&tempMilli, start, end);
        *milliseconds += tempMilli;
        cudaEventDestroy(start);
        cudaEventDestroy(end);

        // Check for errors during hashKernel execution
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("hashKernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
            exit(1);
        }

        Digest *destination = *h_results;
        // Device to host copy

        cudaMemcpyAsync(&(destination[currentIndex]), d_results,
                   sizeof(Digest) * batchSize, cudaMemcpyDeviceToHost, stream1);

        currentIndex += batchSize;
        passwordRemaining -= batchSize;

        // Cleanup before next loop to free memory
        cudaFree(d_passwords);
        cudaFree(d_results);
    }
}

__host__ void chainKernel(int passwordNumber, int numberOfPass, int batchSize, float *milliseconds,
                          Password ** h_passwords, Digest ** h_results, int threadPerBlock,
                          int chainLength) {

    // Device copies for endpoints
    Digest *d_results;
    Password *d_passwords;

    // Measure GPU time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaStream_t stream1;

    cudaStreamCreate(&stream1);

    int chainsRemaining = passwordNumber;
    int currentIndex = 0;

    // Main loop, we add +1 to be sure to do all the batches in case
    // we have 2.5 for example, it'll be 3 passes
    for (int i = 0; i<numberOfPass; i++) {
        // Temporary variable to measure GPU time inside this loop
        float tempMilli = 0;

        // If the currentIndex to save result is greater than the number of
        // password we must stop
        if (currentIndex >= passwordNumber) break;

        // If we have less than batchSize password to hash, then hash them all
        // but modify the batchSize to avoid index errors
        if (chainsRemaining <= batchSize) batchSize = chainsRemaining;

        // GPU Malloc for the password array, size is batchSize
        cudaMalloc(&d_passwords, sizeof(Password) * batchSize);
        cudaMalloc(&d_results, sizeof(Digest) * batchSize);

        Password *source = *h_passwords;

        // Device copies
        cudaMemcpyAsync(d_passwords, &(source[currentIndex]), sizeof(Password) * batchSize,
                        cudaMemcpyHostToDevice, stream1);

        cudaEventRecord(start);
        ntlm_chain_kernel2<<<((batchSize) / threadPerBlock), threadPerBlock, 0, stream1>>>(
                d_passwords, d_results, chainLength);
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        // Necessary procedure to record time and store the elasped time in
        // tempMilli
        cudaEventElapsedTime(&tempMilli, start, end);
        *milliseconds += tempMilli;
        cudaEventDestroy(start);
        cudaEventDestroy(end);

        // Check for errors during hashKernel execution
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("hashKernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
            exit(1);
        }

        Digest *destination = *h_results;
        // Device to host copy

        cudaMemcpyAsync(&(destination[currentIndex]), d_results,
                        sizeof(Digest) * batchSize, cudaMemcpyDeviceToHost, stream1);

        currentIndex += batchSize;
        chainsRemaining -= batchSize;

        // Cleanup before next loop to free memory
        cudaFree(d_passwords);
        cudaFree(d_results);
    }
}

__host__ void printDigest(Digest * dig) {

    for(unsigned char byte : dig->bytes){
        printf("%X02", byte);
    }

    printf("\n");
}

__host__ void printPassword(Password * pwd) {
    for(unsigned char byte : pwd->bytes){
        printf("%c", byte);
    }
    printf("\n");
}