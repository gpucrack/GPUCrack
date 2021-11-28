/*
 * Author: Maxime Missichini
 * Email: missichini.maxime@gmail.com
 * -----
 * File: parallelized_hash.cu
 * Created Date: 28/09/2021
 * -----
 *
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "constants.cuh"
#include "myMd5.cuh"
#include "ntlm.cuh"

int main() {

    size_t freeMem;
    size_t totalMem;
    cudaError_t mem = cudaMemGetInfo(&freeMem, &totalMem);
    freeMem -= 1000000000;
    if (mem != cudaSuccess) {
        printf("memory check failed with error \"%s\".\n",
               cudaGetErrorString(mem));
        return 1;
    }

    printf("MEMORY AVAILABLE : %ld Megabytes\n",(freeMem/1000000));

    size_t memResult = sizeof(Digest) * PASSWORD_NUMBER;
    size_t memPasswords = sizeof(Password) * PASSWORD_NUMBER;
    size_t memUsed = memPasswords + memResult;

    printf("MEMORY USED BY RESULT ARRAY : %ld Megabytes\n",(memResult/1000000));
    printf("MEMORY USED BY PASSWORD ARRAY : %ld Megabytes\n",(memPasswords/1000000));

    printf("THIS MUCH MEMORY WILL BE USED : %ld Megabytes\n",(memUsed/1000000));

    if (memResult > freeMem) {
        printf("NOT ENOUGHT MEMORY TO STORE RESULTS\n");
        exit(1);
    }else if ((double)memResult > 0.65*(double)freeMem) {
        printf("NOT ENOUGHT MEMORY BE OPTIMAL, LOWER PASSWORD_NUMBER\n");
        exit(1);
    }

    size_t remainingMemory = freeMem - memResult;

    printf("MEMORY OPTIMISABLE : %ld Megabytes\n", remainingMemory/1000000);

    auto numberOfPass = (double)((double)memPasswords/(double)remainingMemory);
    int batchSize;

    printf("NUMBER OF PASS : %f\n", numberOfPass);

    const BYTE test_password[7] = {'1', '2', '3', '4', '5', '6', '7'};

    // result = ((number + multiple/2) / multiple) * multiple;
    if (numberOfPass > 1) batchSize = ((((int)(PASSWORD_NUMBER / numberOfPass)) + 1) / 2) * 2;
    else batchSize = PASSWORD_NUMBER;
    int passwordRemaining = PASSWORD_NUMBER;
    int currentIndex = 0;
    printf("FIRST BATCH SIZE : %d\n", batchSize);

    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    // Host copies
    Password *passwords_to_hash;
    //Password *file_buffer;

    Digest *d_results;
    cudaMalloc(&d_results, sizeof(Digest) * PASSWORD_NUMBER);

    Password *d_passwords;

    // Mesure time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float milliseconds = 0;

    for(long i=0; i< (int)numberOfPass+1; i++) {

        float tempMilli = 0;

        // We store everything inside arrays of pointers to char pointers into host
        // memory first
        passwords_to_hash = (Password *)malloc(sizeof(Password) * batchSize);
        //file_buffer = (Password *)malloc(sizeof(Password) * PASSWORD_NUMBER);

        // Opening the file with passwords to hash
        // FILE *fp = fopen("passwords.txt", "r");

        // if (fp == nullptr) {
        //     perror("Error while opening the file\n");
        //     exit(EXIT_FAILURE);
        //}

        for (int n=0; n<batchSize; n++) {
            // fgets((char*)file_buffer[n],MAX_PASSWORD_LENGTH,fp);
            strcpy((char *)passwords_to_hash[n].bytes, (char *)test_password);

            // TO TEST INPUTS
            // printf("%s\n",file_buffer[n]);
        }

        // printf("PASSWORD FILE TO BUFFER DONE @ %f seconds\n",
        //       (double)(clock() - program_start) / CLOCKS_PER_SEC);

        // Simple copy
        // for (int i = 0; i < PASSWORD_NUMBER; i++) {
        //    cudaMemcpy(passwords_to_hash[i], file_buffer[i],
        //               PASSWORD_LENGTH * sizeof(BYTE), cudaMemcpyHostToDevice);
        //}

        // fclose(fp);

        cudaMalloc(&d_passwords, sizeof(Password) * batchSize);

        // Device copies
        cudaMemcpy(d_passwords, passwords_to_hash,
                   sizeof(Password) * batchSize, cudaMemcpyHostToDevice);

        //printf("COPY TO GPU DONE @ %f seconds\n",
        //       (double)(clock() - program_start) / CLOCKS_PER_SEC);

        if (currentIndex >= PASSWORD_NUMBER) break;
        if (passwordRemaining < batchSize) batchSize = passwordRemaining;
        //printf("PASSWORD REMAINING : %d, BATCH SIZE : %d\n", passwordRemaining, batchSize);
        //printf("CURRENT INDEX : %d\n", currentIndex);
        cudaEventRecord(start);
        ntlm<<<batchSize / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(d_passwords, d_results, currentIndex);
        cudaEventRecord(end);

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

        if (i == 0) currentIndex += batchSize-1;
        else currentIndex += batchSize;
        passwordRemaining -= batchSize;
        //printf("NEW CURRENT INDEX : %d\n", currentIndex);

        free(passwords_to_hash);
        cudaFree(d_passwords);
    }

    printf("KERNEL DONE @ %f seconds\n",
           (double)(clock() - program_start) / CLOCKS_PER_SEC);

    Digest *results;
    results = (Digest *)malloc(PASSWORD_NUMBER * sizeof(Digest));

    // Copy back the device result array to host result array
    cudaMemcpy(results, d_results, sizeof(Digest *) * PASSWORD_NUMBER,
               cudaMemcpyDeviceToHost);

    printf("HASH RETRIEVED @ %f seconds\n",
           (double)(clock() - program_start) / CLOCKS_PER_SEC);

    printf("SAMPLE OF OUTPUT : ");
    for (int i = 0; i < HASH_LENGTH; i++) {
        printf("%x", results[666].bytes[i]);
    }
    printf("\n");

    printf("GPU PARALLEL HASH TIME : %f seconds\n", milliseconds / 1000);
    printf("HASH RATE : %f MH/s\n",
           (PASSWORD_NUMBER / (milliseconds / 1000)) / 1000000);

    // Cleanup
    free(results);
    //free(file_buffer);
    cudaFree(d_results);

    program_end = clock();
    program_time_used =
        ((double)(program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);

    return 0;
}
