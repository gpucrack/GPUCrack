#include "hash.cuh"

void hash(Password *h_passwords, Digest *h_results, int passwordNumber, int numberOfPass, bool noPrint) {

    int batchSize = computeBatchSize(numberOfPass, passwordNumber);

    // Measure global time
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    float milliseconds = 0;

    hashKernel(numberOfPass, batchSize, &milliseconds, &program_start, &h_results, &h_passwords, passwordNumber,
               THREAD_PER_BLOCK);

    if(!noPrint) {
        // Compute GPU time and hash rate
        printf("GPU PARALLEL HASH TIME : %f milliseconds\n", milliseconds);
        printf("HASH RATE : %f MH/s\n",
               (passwordNumber / (milliseconds / 1000)) / 1000000);

        // End and compute total time
        program_end = clock();
        program_time_used =
                ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
        printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);
    }
}


// Another version using a time variable, so we can retrieve its value
void hashTime(Password *h_passwords, Digest * h_results, int passwordNumber, float *milliseconds,
              int threadPerBlock, int numberOfPass) {

    int batchSize = computeBatchSize(numberOfPass, passwordNumber);

    // Measure global time
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    hashKernel(numberOfPass, batchSize, milliseconds, &program_start, &h_results, &h_passwords, passwordNumber,
               threadPerBlock);

    // Compute GPU time and hash rate
    printf("GPU PARALLEL HASH TIME : %f milliseconds\n", *milliseconds);
    printf("HASH RATE : %f MH/s\n",
           (passwordNumber / (*milliseconds / 1000)) / 1000000);

    // End and compute total time
    program_end = clock();
    program_time_used =
            ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);
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

        if (batchSize < threadPerBlock) {
            threadPerBlock = batchSize;
        }

        cudaEventRecord(start);
        ntlm_kernel<<<((batchSize) / threadPerBlock), threadPerBlock, 0, stream1>>>(
                d_passwords, d_results);
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        // Necessary procedure to record time and store the elapsed time in
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

    cudaStreamDestroy(stream1);
}