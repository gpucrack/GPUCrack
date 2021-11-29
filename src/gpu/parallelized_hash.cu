#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "constants.cuh"
#include "myMd5.cuh"
#include "ntlm.cuh"

int main() {

    // Checking available memory on the device, store free memory into freeMem
    // and total memory into totalMem
    size_t freeMem;
    size_t totalMem;
    cudaError_t mem = cudaMemGetInfo(&freeMem, &totalMem);

    // Just to keep a little bit of memory, just in case
    freeMem -= 500000000;

    // Checking errors on memory detection
    if (mem != cudaSuccess) {
        printf("memory check failed with error \"%s\".\n",
               cudaGetErrorString(mem));
        return 1;
    }

    printf("MEMORY AVAILABLE : %ld Megabytes\n",(freeMem/1000000));

    // Computing memory used by password and result array
    size_t memResult = sizeof(Digest) * PASSWORD_NUMBER;
    size_t memPasswords = sizeof(Password) * PASSWORD_NUMBER;
    size_t memUsed = memPasswords + memResult;

    printf("MEMORY USED BY RESULT ARRAY : %ld Megabytes\n",(memResult/1000000));
    printf("MEMORY USED BY PASSWORD ARRAY : %ld Megabytes\n",(memPasswords/1000000));

    printf("THIS MUCH MEMORY WILL BE USED : %ld Megabytes\n",(memUsed/1000000));

    // If the result array is greater than the available memory, it's useless to try
    if (memResult > freeMem) {
        printf("NOT ENOUGHT MEMORY TO STORE RESULTS\n");
        exit(1);

        // Then we try to check if at least 65% of the memory is available for the passwords
        // to lower the number of batch we have to do
    }else if ((double)memResult > 0.65*(double)freeMem) {
        printf("NOT ENOUGHT MEMORY BE OPTIMAL, LOWER PASSWORD_NUMBER\n");
        exit(1);
    }

    // We need  to compute how much memory we can use for passwords
    size_t remainingMemory = freeMem - memResult;

    printf("MEMORY OPTIMIZABLE : %ld Megabytes\n", remainingMemory/1000000);

    // We need to determine how many batch we'll do to hash all passwords
    // We need to compute the batch size as well
    auto numberOfPass = (double)((double)memPasswords/(double)remainingMemory);
    int batchSize;

    printf("NUMBER OF PASS : %f\n", numberOfPass);

    // We use a constant password for now
    const BYTE test_password[7] = {'1', '2', '3', '4', '5', '6', '7'};

    // Formula to round down is : result = ((number + multiple/2) / multiple) * multiple;
    if (numberOfPass > 1) batchSize = ((((int)(PASSWORD_NUMBER / numberOfPass)) + 1) / 2) * 2;

    // If we have less than 1 round then the batch size is the number of passwords
    else batchSize = PASSWORD_NUMBER;

    int passwordRemaining = PASSWORD_NUMBER;
    int currentIndex = 0;
    printf("FIRST BATCH SIZE : %d\n", batchSize);

    // Measure global time
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    // Host copies
    Password *passwords_to_hash;
    //Password *file_buffer;

    // Device copies
    Digest *d_results;
    cudaMalloc(&d_results, sizeof(Digest) * PASSWORD_NUMBER);

    Password *d_passwords;

    // Measure GPU time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float milliseconds = 0;

    // Main loop, we add +1 to be sure to do all the batches in case
    // we have 2.5 for example, it'll be 3 passes
    for(long i=0; i< (int)numberOfPass+1; i++) {

        // Temporary variable to measure GPU time inside this loop
        float tempMilli = 0;

        // We store everything inside a big array into host memory first
        passwords_to_hash = (Password *)malloc(sizeof(Password) * batchSize);
        // file_buffer = (Password *)malloc(sizeof(Password) * PASSWORD_NUMBER);

        // Opening the file with passwords to hash
        // FILE *fp = fopen("passwords.txt", "r");

        // Checking for errors when opening the file
        // if (fp == nullptr) {
        //     perror("Error while opening the file\n");
        //     exit(EXIT_FAILURE);
        // }

        for (int n=0; n<batchSize; n++) {
            // Reading lines from the file
            // fgets((char*)file_buffer[n],MAX_PASSWORD_LENGTH,fp);

            // Copying the constant password
            strcpy((char *)passwords_to_hash[n].bytes, (char *)test_password);

            // To test inputs
            // printf("%s\n",file_buffer[n]);
        }

        // Copy data from file buffer reader to password array
        // for (int i = 0; i < PASSWORD_NUMBER; i++) {
        //    cudaMemcpy(passwords_to_hash[i], file_buffer[i],
        //               PASSWORD_LENGTH * sizeof(BYTE), cudaMemcpyHostToDevice);
        //}

        // Close the file
        // fclose(fp);

        // GPU Malloc for the password array, size is batchSize
        cudaMalloc(&d_passwords, sizeof(Password) * batchSize);

        // Device copies
        cudaMemcpy(d_passwords, passwords_to_hash,
                   sizeof(Password) * batchSize, cudaMemcpyHostToDevice);

        // If the currentIndex to save result is greater than the number of password
        // we must stop
        if (currentIndex >= PASSWORD_NUMBER) break;

        // If we have less than batchSize password to hash, then hash them all
        // but modify the batchSize to avoid index errors
        if (passwordRemaining < batchSize) batchSize = passwordRemaining;

        // Debug print
        //printf("PASSWORD REMAINING : %d, BATCH SIZE : %d\n", passwordRemaining, batchSize);
        //printf("CURRENT INDEX : %d\n", currentIndex);

        // Measure time before and after kernel launch
        cudaEventRecord(start);
        ntlm<<<batchSize / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(d_passwords, d_results, currentIndex);
        cudaEventRecord(end);

        // Necessary procedure to record time and store the elasped time in tempMilli
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&tempMilli, start, end);
        milliseconds += tempMilli;

        // Check for errors during kernel execution
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("kernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
            return 1;
        }

        // Fix the index because array begin at index 0, not 1
        if (i == 0) currentIndex += batchSize-1;
        // If we don't have to fix it then just add batchSize
        else currentIndex += batchSize;
        passwordRemaining -= batchSize;

        // Debug
        // printf("NEW CURRENT INDEX : %d\n", currentIndex);

        // Cleanup before next loop to free memory
        free(passwords_to_hash);
        cudaFree(d_passwords);
    }

    printf("KERNEL DONE @ %f seconds\n",
           (double)(clock() - program_start) / CLOCKS_PER_SEC);

    // Host copies
    Digest *results;
    results = (Digest *)malloc(PASSWORD_NUMBER * sizeof(Digest));

    // Copy back the device result array to host result array
    cudaMemcpy(results, d_results, sizeof(Digest *) * PASSWORD_NUMBER,
               cudaMemcpyDeviceToHost);

    printf("HASH RETRIEVED @ %f seconds\n",
           (double)(clock() - program_start) / CLOCKS_PER_SEC);

    // Debug
    printf("SAMPLE OF OUTPUT : ");
    for (int i = 0; i < HASH_LENGTH; i++) {
        printf("%x", results[666].bytes[i]);
    }
    printf("\n");

    // Compute GPU time and hash rate
    printf("GPU PARALLEL HASH TIME : %f seconds\n", milliseconds / 1000);
    printf("HASH RATE : %f MH/s\n",
           (PASSWORD_NUMBER / (milliseconds / 1000)) / 1000000);

    // Cleanup
    free(results);
    //free(file_buffer);
    cudaFree(d_results);

    // End and compute total time
    program_end = clock();
    program_time_used =
        ((double)(program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);

    return 0;
}
