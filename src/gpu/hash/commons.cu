#include "commons.cuh"

__host__ Password * generatePasswords(int passwordNumber) {

    auto * result = (Password*) malloc(passwordNumber*sizeof(Password));

    char charSet[62] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x',
                      'y','z','1','2','3','4','5','6','7','8','9','0','A','B','C','D','E','F','G','H','I','J','K'
    ,'L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 61); // define the range

    printf("\n==========GENERATING PASSWORDS==========\n");
    // Generate all passwords
    for(int j=0; j<passwordNumber; j++) {
        auto * currentPassword = (Password *) malloc(sizeof(Password));
        // Generate one password
        for (unsigned char &byte: (*currentPassword).bytes) {
            byte = charSet[distr(gen)];
        }

        result[j] = *(currentPassword);
        free(currentPassword);
    }
    printf("DONE, %d PASSWORDS GENERATED\n", passwordNumber);
    printf("====================\n");

    return result;
}

// Returns the number of batch that we need to do
__host__ double memoryAnalysis(int passwordNumber) {

    printf("\n==========MEMORY ANALYSIS==========\n");

    // Checking if THREAD_PER_BLOCK is a power of 2 because we will have memory problems otherwise
    if ((ceil(log2(THREAD_PER_BLOCK)) != floor(log2(THREAD_PER_BLOCK)))){
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
    auto numberOfPass = (double)((double)memUsed / (double)freeMem);

    printf("NUMBER OF PASS : %f\n", numberOfPass);

    printf("====================\n");

    return numberOfPass;
}

__host__ int computeBatchSize(double initialNumberOfPass, int passwordNumber) {
    int batchSize;

    // Formula to round down is : result = ((number + multiple/2) / multiple) *
    // multiple;
    initialNumberOfPass += 0.5;
    int numberOfPass = (int)initialNumberOfPass;
    if ((numberOfPass % 2) != 0) numberOfPass++;
    printf("%d\n", (int)numberOfPass);


    if ((int)numberOfPass > 1)
        batchSize = ((int)(passwordNumber / (int)numberOfPass));
    // If we have less than 1 round then the batch size is the number of
    // passwords
    else
        batchSize = passwordNumber;

    return batchSize;
}

__host__ void kernel(const double numberOfPass, int batchSize,
                     float *milliseconds, const clock_t *program_start,
                     Digest **h_results, Password **h_passwords, int passwordNumber) {

    printf("\n==========LAUNCHING KERNEL==========\n");

    *h_results = (Digest *)malloc(passwordNumber * sizeof(Digest));

    // Device copies
    Digest *d_results;
    Password *d_passwords;

    // Measure GPU time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int passwordRemaining = passwordNumber;
    int currentIndex = 0;

    printf("FIRST BATCH SIZE : %d\n", batchSize);

    // Main loop, we add +1 to be sure to do all the batches in case
    // we have 2.5 for example, it'll be 3 passes
    for (int i = 0; i < (int)numberOfPass + 1; i++) {
        // Temporary variable to measure GPU time inside this loop
        float tempMilli = 0;

        // GPU Malloc for the password array, size is batchSize
        cudaMalloc(&d_passwords, sizeof(Password) * batchSize);
        cudaMalloc(&d_results, sizeof(Digest) * batchSize);

        // If the currentIndex to save result is greater than the number of
        // password we must stop
        if (currentIndex >= passwordNumber) break;

        // If we have less than batchSize password to hash, then hash them all
        // but modify the batchSize to avoid index errors
        if (passwordRemaining <= batchSize) batchSize = passwordRemaining;
        printf("BATCHSIZE: %d\n", batchSize);

        // Debug print
        // printf("PASSWORD REMAINING : %d, BATCH SIZE : %d\n",
        // passwordRemaining, batchSize); printf("CURRENT INDEX : %d\n",
        // currentIndex);

        Password *source = *h_passwords;
        // Device copies
        cudaMemcpy(d_passwords, &(source[currentIndex]), sizeof(Password) * batchSize,
                   cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        ntlm_kernel<<<((batchSize) / THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            d_passwords, d_results);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        // Necessary procedure to record time and store the elasped time in
        // tempMilli
        cudaEventElapsedTime(&tempMilli, start, end);
        *milliseconds += tempMilli;
        cudaEventDestroy(start);
        cudaEventDestroy(end);

        printf("KERNEL #%d DONE @ %f seconds\n", i,
               (double)(clock() - *program_start) / CLOCKS_PER_SEC);

        // Check for errors during kernel execution
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("kernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
            exit(1);
        }

        Digest *destination = *h_results;
        // Device to host copy
        cudaMemcpy(&(destination[currentIndex]), d_results,
                   sizeof(Digest) * batchSize, cudaMemcpyDeviceToHost);

        // Fix the index because array begin at index 0, not 1
        if (i == 0) currentIndex += batchSize - 1;
        // If we don't have to fix it then just add batchSize
        else
            currentIndex += batchSize;
        passwordRemaining -= batchSize;

        printf("CURRENT INDEX: %d\n", currentIndex);
        // Debug
        // printf("NEW CURRENT INDEX : %d\n", currentIndex);

        // Cleanup before next loop to free memory
        cudaFree(d_passwords);
        cudaFree(d_results);
    }
    printf("====================\n");
}