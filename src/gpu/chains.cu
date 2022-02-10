#include "chains.cuh"

__device__ static const unsigned char charset[CHARSET_LENGTH] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
                                                     'D', 'E',
                                                     'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                                                     'S', 'T',
                                                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                                                     'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                                     'q', 'r',
                                                     's', 't',
                                                     'u', 'v', 'w', 'x', 'y', 'z'};

__host__ void
generateChains(Password *h_passwords, Digest *h_results, int passwordNumber, int numberOfPass, int numberOfColumn,
               bool save, int theadsPerBlock, bool debug) {

    if (debug) {
        printf("\n==========INPUTS==========\n");
        for (int i = passwordNumber - 1; i < passwordNumber; i++) {
            printPassword(&(h_passwords[i]));
        }
        printf("\n");
    }

    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    float milliseconds = 0;

    int batchSize = computeBatchSize(numberOfPass, passwordNumber);

    // We send numberOfColumn/2 since one loop of kernel is hashing/reducing at the same time so we need 2x
    // less operations
    chainKernel(passwordNumber, numberOfPass, batchSize, &milliseconds,
                &h_passwords, &h_results, theadsPerBlock,
                numberOfColumn / 2, save, false);

    if (debug) {
        printf("Total GPU time : %f milliseconds\n", milliseconds);
        printf("Chain rate : %f MC/s\n",
               ((float) (passwordNumber) / (milliseconds / 1000)) / 1000000);
        printf("Hash/Reduction : %f MHR/s\n",
               (((float) (passwordNumber) / (milliseconds / 1000)) / 1000000) * (float) numberOfColumn);

        program_end = clock();
        program_time_used =
                ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
        printf("Total execution time : %f seconds\n", program_time_used);
    }

    if (debug) {
        printf("\n==========OUTPUTS==========\n");
        for (int i = passwordNumber - 1; i < passwordNumber; i++) {
            printPassword(&(h_passwords[i]));
        }
        printf("\n");
    }

}

__device__ void reduce_digest(unsigned int index, Digest *digest, Password *plain_text) {
    (*plain_text).i[0] =
            charset[((*digest).bytes[0] + index) % CHARSET_LENGTH] |
            (charset[((*digest).bytes[1] + index) % CHARSET_LENGTH] << 8)|
            (charset[((*digest).bytes[2] + index) % CHARSET_LENGTH] << 16)|
            (charset[((*digest).bytes[3] + index) % CHARSET_LENGTH] << 24);
    (*plain_text).i[1] =
            charset[((*digest).bytes[4] + index) % CHARSET_LENGTH] |
            (charset[((*digest).bytes[5] + index) % CHARSET_LENGTH] << 8)|
            (charset[((*digest).bytes[6] + index) % CHARSET_LENGTH] << 16)|
            (charset[((*digest).bytes[7] + index) % CHARSET_LENGTH] << 24);
}

__device__ void progressBar(double percentage) {
    if((percentage * 10) % 10 )
    unsigned char x = 219;
    cout << x;
}


void initLoadingBar() {
    // Initialize chars for printing loading bar
    unsigned char a = 177, b = 219;

    printf("\n\n\n\n");
    printf("\n\n\n\n\t\t\t\t\t"
           + "Generating table...\n\n");
    printf("\t\t\t\t\t");

    // Print initial loading bar
    for (int i = 0; i < 20; i++)
        printf("%c", a);
}

void incrementLoadingBar() {
    // Set the cursor again starting
    // point of loading bar
    printf("\r");
    printf("\t\t\t\t\t");

    printf("%c", b);
}

__global__ void ntlm_chain_kernel(Password *passwords, Digest *digests, int chainLength) {

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    double progress = 1.;

    for (int i = 0; i < chainLength; i++) {

        if (index == 0) {
            if (((double)i/(double)chainLength) > progress/10) incrementLoadingBar(progress);
            progress += 1.;
        }

        ntlm(&passwords[index], &digests[index]);
        reduce_digest(i, &digests[index], &passwords[index]);
    }
}

__host__ void
chainKernel(int passwordNumber, int numberOfPass, int batchSize, float *milliseconds, Password **h_passwords,
            Digest **h_results, int threadPerBlock, int chainLength, bool save, bool debug) {

    if (save) {
        createFile((char *) "../src/tables/testStart.txt", true);
        writePoint((char *) "../src/tables/testStart.txt", h_passwords, passwordNumber
                   , true);
    }

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

    printf("Generating chains...\n\n");

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
        cudaMalloc(&d_passwords, sizeof(Password) * batchSize);
        cudaMalloc(&d_results, sizeof(Digest) * batchSize);

        Password *source = *h_passwords;

        // Device copies
        cudaMemcpyAsync(d_passwords, &(source[currentIndex]), sizeof(Password) * batchSize,
                        cudaMemcpyHostToDevice, stream1);

        cudaEventRecord(start);
        ntlm_chain_kernel<<<((batchSize) / threadPerBlock), threadPerBlock, 0, stream1>>>(
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
    cudaStreamDestroy(stream1);

    if (save) {
        createFile((char *) "../src/tables/testEnd.txt", true);
        writePoint((char *) "../src/tables/testEnd.txt", h_passwords, passwordNumber,
                   true);
    }
}