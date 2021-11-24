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

    printf("MEMORY AVAILABLE : %ld Megabytes\n",(totalMem/1000000));

    size_t memUsed = sizeof(Password) * PASSWORD_NUMBER + sizeof(Digest) * PASSWORD_NUMBER;

    printf("THIS MUCH MEMORY WILL BE USED : %ld Megabytes\n",(memUsed/1000000));

    auto numberOfPass = (double)((double)memUsed/(double)totalMem);

    printf("NUMBER OF PASS : %f\n", numberOfPass);


    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    // Host copies
    Password *passwords_to_hash;
    Password *file_buffer;

    // We store everything inside arrays of pointers to char pointers into host
    // memory first
    passwords_to_hash = (Password *)malloc(sizeof(Password) * PASSWORD_NUMBER);
    file_buffer = (Password *)malloc(sizeof(Password) * PASSWORD_NUMBER);

    // Opening the file with passwords to hash
    // FILE *fp = fopen("passwords.txt", "r");

    // if (fp == nullptr) {
    //     perror("Error while opening the file\n");
    //     exit(EXIT_FAILURE);
    //}

    const BYTE test_password[7] = {'1', '2', '3', '4', '5', '6', '7'};
    int n = 0;
    while (n < PASSWORD_NUMBER) {
        // fgets((char*)file_buffer[n],MAX_PASSWORD_LENGTH,fp);
        strcpy((char *)passwords_to_hash[n].bytes, (char *)test_password);

        // TO TEST INPUTS
        // printf("%s\n",file_buffer[n]);
        n++;
    }

    // printf("PASSWORD FILE TO BUFFER DONE @ %f seconds\n",
    //       (double)(clock() - program_start) / CLOCKS_PER_SEC);

    // Simple copy
    // for (int i = 0; i < PASSWORD_NUMBER; i++) {
    //    cudaMemcpy(passwords_to_hash[i], file_buffer[i],
    //               PASSWORD_LENGTH * sizeof(BYTE), cudaMemcpyHostToDevice);
    //}

    // fclose(fp);

    Password *d_passwords;
    cudaMalloc(&d_passwords, sizeof(Password) * PASSWORD_NUMBER);

    Digest *d_results;
    cudaMalloc(&d_results, sizeof(Digest) * PASSWORD_NUMBER);

    // Device copies
    cudaMemcpy(d_passwords, passwords_to_hash,
               sizeof(Password) * PASSWORD_NUMBER, cudaMemcpyHostToDevice);

    printf("COPY TO GPU DONE @ %f seconds\n",
           (double)(clock() - program_start) / CLOCKS_PER_SEC);

    // Mesure time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    ntlm<<<PASSWORD_NUMBER / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(d_passwords, d_results);
    //kernel_md5_hash<<<PASSWORD_NUMBER / (THREAD_PER_BLOCK), (THREAD_PER_BLOCK)>>>(d_passwords,d_results);
    cudaEventRecord(end);

    // Check for errors during kernel execution
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
        return 1;
    }

    printf("KERNEL DONE @ %f seconds\n",
           (double)(clock() - program_start) / CLOCKS_PER_SEC);

    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

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
    free(passwords_to_hash);
    free(results);
    free(file_buffer);
    cudaFree(d_passwords);
    cudaFree(d_results);

    program_end = clock();
    program_time_used =
        ((double)(program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);

    return 0;
}
