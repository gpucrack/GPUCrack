#include <cstdio>
#include <cstdlib>

#include "../constants.cuh"
#include "../hash_functions/cudaMd5.cuh"
#include "test_hash.cuh"

#define REFERENCE_SENTENCE1 "The quick brown fox jumps over the lazy dog"
#define REFERENCE_RESULT1 "9e107d9d372bb6826bd81d3542a419d6"
#define NUMBER_OF_PASSWORD 1

int main() {
    BYTE **passwords = (BYTE **)malloc(NUMBER_OF_PASSWORD * sizeof(BYTE *));
    BYTE **results = (BYTE **)malloc(NUMBER_OF_PASSWORD * sizeof(BYTE *));

    for (int i = 0; i < NUMBER_OF_PASSWORD; i++) {
        // Each time we allocate the host pointer into device memory
        cudaMalloc((void **)&passwords[i], PASSWORD_LENGTH * sizeof(BYTE));
        cudaMalloc((void **)&results[i], PASSWORD_LENGTH * sizeof(BYTE));
    }

    int count;
    for (count = 0; count < PASSWORD_LENGTH; count++) {
        if (REFERENCE_SENTENCE1[count] == '\0') {
            printf("PASSWORD LENGTH : %d\n", count);
            break;
        }
    }
    cudaMemcpy(passwords[0], REFERENCE_SENTENCE1, count * sizeof(BYTE),
               cudaMemcpyHostToDevice);

    WORD total_length = count;
    WORD length = count;

    BYTE **d_passwords;
    BYTE **d_results;
    WORD *d_total_length;
    WORD *d_length;

    cudaMalloc((void **)&d_passwords, sizeof(BYTE *) * NUMBER_OF_PASSWORD);
    cudaMalloc((void **)&d_results, sizeof(BYTE *) * NUMBER_OF_PASSWORD);
    cudaMalloc((void **)&d_total_length, sizeof(WORD *));
    cudaMalloc((void **)&d_length, sizeof(WORD *));
    cudaMemcpy(d_passwords, passwords, sizeof(BYTE *) * NUMBER_OF_PASSWORD,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_results, results, sizeof(BYTE *) * NUMBER_OF_PASSWORD,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_length, &total_length, sizeof(WORD),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_length, &length, sizeof(WORD), cudaMemcpyHostToDevice);

    CUDA_MD5_CTX context;
    kernel_md5_hash<<<NUMBER_OF_PASSWORD, 1>>>(d_passwords, d_total_length,
                                               d_results, d_length, context);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
        return 1;
    }

    // Then, in order to copy back we need a host array of pointers to char
    // pointers
    BYTE **final_results;
    final_results = (BYTE **)malloc(NUMBER_OF_PASSWORD * sizeof(BYTE *));

    // We need to allocate each char pointers
    for (int k = 0; k < NUMBER_OF_PASSWORD; k++) {
        final_results[k] = (BYTE *)malloc(PASSWORD_LENGTH * sizeof(BYTE));
    }

    // Copy back the device result array to host result array
    cudaMemcpy(results, d_results, sizeof(BYTE *) * NUMBER_OF_PASSWORD,
               cudaMemcpyDeviceToHost);

    int j;
    // Deep copy of each pointers to the host result array
    for (j = 0; j < NUMBER_OF_PASSWORD; j++) {
        cudaMemcpy(final_results[j], results[j], PASSWORD_LENGTH * sizeof(BYTE),
                   cudaMemcpyDeviceToHost);
    }
    printf("PASSWORD RETRIEVED : %d\n", j);

    bool test1 = strcmp((char *)final_results[0], REFERENCE_RESULT1);

    printf("RESULTS FROM CUDA FUNCTION : ");
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        if (final_results[0][i] == '\0') break;
        printf("%x", final_results[0][i]);
    }

    printf("\n");

    // Comparing with CPU version
    BYTE buf[HASH_LENGTH];
    MD5_CTX ctx;

    md5_init(&ctx);
    md5_update(&ctx, (BYTE *)REFERENCE_SENTENCE1, strlen(REFERENCE_SENTENCE1));
    md5_final(&ctx, buf);

    printf("RESULTS FROM BASE FUNCTION : ");
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        if (buf[i] == '\0') break;
        printf("%x", buf[i]);
    }
    printf("\n");

    if (test1)
        printf("TEST PASSED !\n");
    else
        printf("TEST FAILED !\n");

    // Cleanup
    free(passwords);
    free(results);
    free(final_results);
    cudaFree(d_passwords);
    cudaFree(d_results);

    return 0;
}