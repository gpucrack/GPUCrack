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

#include "commons.cuh"
#include "rainbow.cuh"

int main() {
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    RainbowTable table;
    table.length = PASSWORD_NUMBER;
    cudaMalloc(&table.chains, sizeof(RainbowChain) * PASSWORD_NUMBER);
    RainbowChain *d_chains = table.chains;

    /*
        See https://stackoverflow.com/a/31135377 for struct allocation with
        device pointers.

        We could also only pass the chains, and pass the length as a separate
        argument.
    */
    RainbowTable *d_table;
    cudaMalloc(&d_table, sizeof(RainbowTable));
    cudaMemcpy(d_table, &table, sizeof(RainbowTable), cudaMemcpyHostToDevice);

    printf("MEMORY ALLOCATED IN GPU IN %f seconds\n",
           (double) (clock() - program_start) / CLOCKS_PER_SEC);

    // Measure time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    ntlm_chain_kernel<<<PASSWORD_NUMBER / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(
            d_table);
    cudaEventRecord(end);

    // Check for errors during kernel execution
    cudaError_t cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaError));
        return 1;
    }

    printf("KERNEL DONE in %f seconds\n",
           (double) (clock() - program_start) / CLOCKS_PER_SEC);

    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    // Copy back the device result array to host result array
    cudaMemcpy(&table, d_table, sizeof(RainbowTable), cudaMemcpyDeviceToHost);

    table.chains =
            (RainbowChain *) malloc(PASSWORD_NUMBER * sizeof(RainbowChain));
    cudaMemcpy(table.chains, d_chains, sizeof(RainbowChain) * PASSWORD_NUMBER,
               cudaMemcpyDeviceToHost);

    printf("CHAINS RETRIEVED IN %f seconds\n",
           (double) (clock() - program_start) / CLOCKS_PER_SEC);

    printf("SAMPLE OF OUTPUT :\n");
    // only show 10 lines
    table.length = 100;
    print_table(&table);

    printf("\n");

    printf("GPU PARALLEL HASH TIME : %f seconds\n", milliseconds / 1000);
    printf("HASH RATE (adjusted with TABLE_T) : %f MH/s\n",
           (PASSWORD_NUMBER / (milliseconds / 1000)) / 1000000 * TABLE_T);

    // Cleanup
    cudaFree(d_table);

    program_end = clock();
    program_time_used =
            ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);

    return 0;
}