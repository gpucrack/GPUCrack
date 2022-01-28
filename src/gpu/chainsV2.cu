#include "chainsV2.cuh"

__device__ static const unsigned char charset[64] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                                                      'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                                                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                                                      'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                                                      's', 't',
                                                      'u', 'v', 'w', 'x', 'y', 'z', '-', '_'};

__host__ void generateChains(Password * h_passwords, Digest * h_results, int passwordNumber, int numberOfPass) {

    printf("\n==========INPUTS==========\n");
    for(int i=passwordNumber-1; i<passwordNumber; i++) {
        printPassword(&(h_passwords[i]));
        printDigest(&(h_results[i]));
    }
    printf("\n");

    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    float milliseconds = 0;

    int batchSize = computeBatchSize(numberOfPass, passwordNumber);

    //TODO : compute t
    int t = 5;

    chainKernel(passwordNumber, numberOfPass, batchSize, &milliseconds,
            &h_passwords, &h_results, THREAD_PER_BLOCK, t);

    //TODO : save endingpoints on disk

    program_end = clock();
    program_time_used =
            ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);

    printf("\n==========OUTPUTS==========\n");
    for(int i=passwordNumber-1; i<passwordNumber; i++) {
        printPassword(&(h_passwords[i]));
        printDigest(&(h_results[i]));
    }
    printf("\n");

}

__device__ void reduce_digest(unsigned int index, Digest * digest, Password  * plain_text) {
    for (int i = 0; i < PASSWORD_LENGTH - 1; i++) {
        (*plain_text).bytes[i] = charset[((*digest).bytes[i] + index) % 64];
    }
}

__global__ void ntlm_chain_kernel2(Password * passwords, Digest * digests, int chainLength) {

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=0; i<chainLength; i++){
        ntlm(&passwords[index], &digests[index]);
        reduce_digest(index ,&digests[index], &passwords[index]);
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

        //TODO: save start points on disk

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

        Password *destination2 = *h_passwords;
        // Device to host copy

        cudaMemcpyAsync(&(destination2[currentIndex]), d_passwords,
                        sizeof(Password) * batchSize, cudaMemcpyDeviceToHost, stream1);

        currentIndex += batchSize;
        chainsRemaining -= batchSize;

        // Cleanup before next loop to free memory
        cudaFree(d_passwords);
        cudaFree(d_results);
    }
}