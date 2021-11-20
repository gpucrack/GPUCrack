#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "myMd5.cu"

int main() {
    double program_time;
    clock_t program_end;
    clock_t program_start = clock();

    // the passwords on the host side
    Password* passwords = (Password*)malloc(sizeof(Password) * PASSWORD_NUMBER);

    // copy in all password a test string
    const char* test_password = "1234567";
    for (unsigned long i = 0; i < PASSWORD_NUMBER; i++) {
        strcpy(passwords[i].chars, test_password);
    }

    // the passwords on the device side
    Password* d_passwords;
    cudaMalloc(&d_passwords, sizeof(Password) * PASSWORD_NUMBER);

    Digest* d_digests;
    cudaMalloc(&d_digests, sizeof(Digest) * PASSWORD_NUMBER);

    // copy the passwords to the device
    cudaMemcpy(d_passwords, passwords, sizeof(Password) * PASSWORD_NUMBER,
               cudaMemcpyHostToDevice);

    printf("Copy to GPU done in %.2lf seconds\n",
           (double)(clock() - program_start) / CLOCKS_PER_SEC);

    // we don't need passwords in RAM anymore
    free(passwords);

    clock_t device_start = clock();

    md5_hash2<<<PASSWORD_NUMBER / 1024, 1024>>>(d_passwords, d_digests);

    // check for errors during the kernel execution
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(status));
        return EXIT_FAILURE;
    }

    printf("Kernel done in %lf seconds\n",
           (double)(clock() - device_start) / CLOCKS_PER_SEC);

    Digest* digests;
    digests = (Digest*)malloc(sizeof(Digest) * PASSWORD_NUMBER);

    // copy back the digests from the device to the host
    cudaMemcpy(digests, d_digests, sizeof(Digest) * PASSWORD_NUMBER,
               cudaMemcpyDeviceToHost);

    printf("Sample of the output: ");
    for (int i = 0; i < MD5_BLOCK_SIZE; i++) {
        printf("%x", digests[1234].bytes[i]);
    }
    printf("\n");

    // cleanup
    free(digests);
    cudaFree(d_passwords);
    cudaFree(d_digests);

    printf("Total execution time: %lf seconds\n",
           (double)(clock() - program_start) / CLOCKS_PER_SEC);

    return EXIT_SUCCESS;
}