#include "generalTest.cuh"

int main() {

    int passwordNumber = DEFAULT_PASSWORD_NUMBER;

    // Simulate when we send password as input
    Password * passwords = generatePasswords(passwordNumber);

    Digest * result;

    cudaError_t status = cudaMallocHost(&result, passwordNumber * sizeof(Digest));
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    auto numberOfPass = memoryAnalysis(passwordNumber);

    parallelized_hash(passwords, result, passwordNumber, numberOfPass);

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return (0);
}