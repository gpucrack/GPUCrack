#include "chainsV2.cuh"

__host__ int createChain() {
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    int t = 0;

    // Measure time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    ntlm_chain_kernel2<<<DEFAULT_PASSWORD_NUMBER / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(
            t);
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

    // Cleanup
    cudaFree(d_table);

    program_end = clock();
    program_time_used =
            ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);
}

__global__ void ntlm_chain_kernel2(int t) {

}

int main() {
    return createChain();
}