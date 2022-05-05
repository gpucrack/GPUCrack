#include "chains.cuh"

__host__ void
generateChains(Password *h_passwords, unsigned long long passwordNumber, int numberOfPass, int numberOfColumn,
               bool save, int theadsPerBlock, bool debug, bool debugKernel, Digest *h_results, int pwd_length,
               char *start_path, char *end_path, float *totalGPU, int batchNumber) {

    printf("Generating chains...\n");

    float milliseconds = 0;

    unsigned long long batchSize = computeBatchSize(numberOfPass, passwordNumber);

    // We send numberOfColumn/2 since one loop of kernel is hashing/reducing at the same time so we need 2x
    // less operations
    chainKernel(passwordNumber, numberOfPass, batchSize, &milliseconds,
                &h_passwords, theadsPerBlock,
                numberOfColumn, debugKernel, &h_results, pwd_length, start_path, end_path, debugKernel);

    printf("Batch %d done in : %f milliseconds (GPU time).\n", batchNumber, milliseconds);

    *totalGPU = *totalGPU + milliseconds;
}

__global__ void
ntlmChainKernel(Password *passwords, Digest *digests, int chainLength, int pwd_length, unsigned long long domain) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < chainLength; i++) {
        ntlm(&passwords[index], &digests[index], pwd_length);
        reduceDigest(i, &digests[index], &passwords[index], pwd_length, domain);
    }
}

__global__ void
ntlmChainKernelDebug(Password *passwords, Digest *digests, int chainLength, int pwd_length, unsigned long long domain) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;

    // Trick to for a working print
    Password * password = (Password*) malloc(sizeof(Password));
    Digest * digest = (Digest*) malloc(sizeof(Digest));
    for (int i = 0; i < chainLength; i++) {
        if(index == (1)){
            printf("%d: ", i);
            printPassword(&passwords[index]);
            printf(" --> ");
        }
        ntlm(&passwords[index], &digests[index], pwd_length);
        if (index == (1)){
            printDigest(&digests[index]);
            printf(" --> ");
        }
        reduceDigest(i, &digests[index], &passwords[index], pwd_length, domain);
        if(index == (1)){
            printPassword(&passwords[index]);
            printf("\n");
        }
    }
    free(password);
    free(digest);
}

__host__ void
chainKernel(unsigned long long passwordNumber, int numberOfPass, unsigned long long batchSize, float *milliseconds,
            Password **h_passwords, int threadPerBlock, int chainLength, bool debug, Digest **h_results,
            int pwd_length, char *start_path, char *end_path, bool kernelDebug) {

    unsigned long long domain = (unsigned long long)pow(CHARSET_LENGTH, pwd_length);

    if (debug) printf("Domain : %lld\n", domain);

    // Device copies for endpoints
    Digest *d_results;
    Password *d_passwords;

    // Measure GPU time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaStream_t stream1;

    cudaStreamCreate(&stream1);

    unsigned long long chainsRemaining = passwordNumber;
    unsigned long long currentIndex = 0;

    if (debug) printf("Generating chains...\n\n");

    // Main loop, we add +1 to be sure to do all the batches in case
    // we have 2.5 for example, it'll be 3 passes
    for (int i = 0; i < numberOfPass; i++) {
        // Temporary variable to measure GPU time inside this loop
        float tempMilli = 0;

        // If the currentIndex to save result is greater than the number of
        // password we must stop
        if (currentIndex >= passwordNumber) break;

        // If we have less than batchSize password to hash, then hash them all
        // but modify the batchSize to avoid index errors
        if (chainsRemaining <= batchSize) batchSize = chainsRemaining;

        // GPU Malloc for the password array, size is batchSize
        handleCudaError(cudaMalloc(&d_passwords, sizeof(Password) * batchSize));
        handleCudaError(cudaMalloc(&d_results, sizeof(Digest) * batchSize));

        Password *source = *h_passwords;

        // Device copies
        handleCudaError(cudaMemcpyAsync(d_passwords, &(source[currentIndex]), sizeof(Password) * batchSize,
                        cudaMemcpyHostToDevice, stream1));

        cudaEventRecord(start);

        if (kernelDebug)
            ntlmChainKernelDebug<<<((unsigned long long)((unsigned long long)(batchSize) / (unsigned long long)threadPerBlock)) + 1, threadPerBlock, 0, stream1>>>(
                    d_passwords, d_results, chainLength, pwd_length, domain);
        else
            ntlmChainKernel<<<((unsigned long long)((unsigned long long)(batchSize) / (unsigned long long)threadPerBlock)) + 1, threadPerBlock, 0, stream1>>>(
                    d_passwords, d_results, chainLength, pwd_length, domain);

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

        if (debug){
            Digest *destination = *h_results;
            // Device to host copy

            handleCudaError(cudaMemcpyAsync(&(destination[currentIndex]), d_results,
                            sizeof(Digest) * batchSize, cudaMemcpyDeviceToHost, stream1));
        }

        Password *destination2 = *h_passwords;
        // Device to host copy
        handleCudaError(cudaMemcpyAsync(&(destination2[currentIndex]), d_passwords,
                        sizeof(Password) * batchSize, cudaMemcpyDeviceToHost, stream1));

        currentIndex += batchSize;
        chainsRemaining -= batchSize;

        // Cleanup before next loop to free memory
        cudaFree(d_passwords);
        cudaFree(d_results);
    }
    cudaStreamDestroy(stream1);
}