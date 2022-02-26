#include "reduction.cuh"

// The character set used for passwords. We declare it in the host scope and in the device scope.
// The character set used for passwords.
__device__ static const unsigned char charset[CHARSET_LENGTH] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                                                      'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                                                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                                                      'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                                                      's', 't',
                                                      'u', 'v', 'w', 'x', 'y', 'z'};

// The character set used for digests (NTLM hashes).
static const unsigned char hashset[DIGEST_CHARSET_LENGTH] = {0x88, 0x46, 0xF7, 0xEA, 0xEE, 0x8F, 0xB1,
                                                             0x17, 0xAD, 0x06, 0xBD, 0xD8, 0x30, 0xB7,
                                                             0x58, 0x6C};

void generate_digests_random(Digest **digests, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = HASH_LENGTH - 1; i >= 0; i--) {
            (*digests)[j].bytes[i] = hashset[rand() % CHARSET_LENGTH];
        }
    }
}

__host__ __device__ void reduceDigest(unsigned int pos, Digest *digest, Password *plain_text, unsigned long domain) {

    // index so that we are inside the right domain
    unsigned long index = ((*digest).value + pos) % domain;

    for(int i=PASSWORD_LENGTH-1; i>=0; i--){
        (*plain_text).bytes[i] = charset[index % (unsigned long)CHARSET_LENGTH];

        // Dividing by index, so we lose a power each time to stay in the correct domain for the next character
        index /= CHARSET_LENGTH;
    }

    /*
    for(int i=0; i<PASSWORD_LENGTH; i++){
        (*plain_text).bytes[i] = charset[((*digest).bytes[i] + index) % CHARSET_LENGTH];
    }*/
}

__global__ void reduceDigests(Digest *digests, Password *plain_texts, int column, unsigned long domain) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    reduceDigest(column, &digests[idx], &plain_texts[idx], domain);
}

int count_duplicates(Password **passwords, bool debug, int passwordNumber) {
    int count = 0;
    for (int i = 0; i < passwordNumber; i++) {
        if (debug) printf("Searching for duplicate of password number %d...\n", i);
        for (int j = i + 1; j < passwordNumber; j++) {
            // Increment count by 1 if duplicate found
            if (memcmp((*passwords)[i].bytes, (*passwords)[j].bytes, PASSWORD_LENGTH) != 0) {
                printf("Found a duplicate : ");
                printPassword(&(*passwords)[i]);
                count++;
            }
        }
    }
    return count;
}

void display_reductions(Digest *digests, Password *passwords, int n) {
    for (int i = 0; i < n; i++) {
        printDigest(&(digests[i]));
        printf(" --> ");
        printPassword(&(passwords[i]));
        printf("\n");
    }
}

__host__ void reduceKernel(int passwordNumber, int numberOfPass, int batchSize, float *milliseconds,
                          Password **h_passwords, Digest **h_results, int threadPerBlock) {
    Password * d_passwords;
    Digest * d_results;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int reductionRemaining = passwordNumber;
    int currentIndex = 0;

    unsigned long domain = pow(CHARSET_LENGTH, PASSWORD_LENGTH);

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
        if (reductionRemaining <= batchSize) batchSize = reductionRemaining;

        // GPU Malloc for the password array, size is batchSize
        cudaMalloc(&d_passwords, sizeof(Password) * batchSize);
        cudaMalloc(&d_results, sizeof(Digest) * batchSize);

        Digest *source = *h_results;

        // Device copies
        cudaMemcpy(d_results, &(source[currentIndex]), sizeof(Digest) * batchSize,
                        cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        // Reduce all those digests into passwords
        reduceDigests<<<((batchSize) / threadPerBlock), threadPerBlock>>>(d_results,
                                                                          d_passwords, 1, domain);

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

        Password *destination2 = *h_passwords;
        // Device to host copy
        cudaMemcpy(&(destination2[currentIndex]), d_passwords,
                        sizeof(Password) * batchSize, cudaMemcpyDeviceToHost);

        currentIndex += batchSize;
        reductionRemaining -= batchSize;

        // Cleanup before next loop to free memory
        cudaFree(d_passwords);
        cudaFree(d_results);
    }
}

__host__ void
reduce(Password *h_passwords, Digest *h_results, int passwordNumber, int numberOfPass, int threadsPerBlock) {
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    // Generate DEFAULT_PASSWORD_NUMBER digests
    printf("Generating digests...\n");
    generate_digests_random(&h_results, passwordNumber);
    printf("Digest generation done!\n");

    float milliseconds = 0;

    int batchSize = computeBatchSize(numberOfPass, passwordNumber);

    reduceKernel(passwordNumber, numberOfPass, batchSize, &milliseconds, &h_passwords, &h_results, threadsPerBlock);

    //display_reductions(h_results, h_passwords, passwordNumber);

    printf("TOTAL GPU TIME : %f milliseconds\n", milliseconds);

    double reduce_rate = ((double)passwordNumber / (milliseconds / 1000)) / 1000000;

    printf("Reduction of %d digests ended after %f milliseconds.\n Reduction rate: %f MR/s.\n", passwordNumber,
           (double) milliseconds, reduce_rate);

    /*
    int dup = count_duplicates(&h_passwords, false, 0);
    printf("Found %d duplicate(s) among the %d reduced passwords (%f percent).\n", dup, passwordNumber,
           ((double) dup / passwordNumber) * 100);*/

    program_end = clock();
    program_time_used =
            ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);
}