#include "chainsV2.cuh"

__host__ int generateChains(Password * h_passwords, Digest * h_results, int passwordNumber, int numberOfPass) {
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    float milliseconds = 0;

    int batchSize = computeBatchSize(numberOfPass, passwordNumber);

    //TODO : compute t
    int t = 0;

    chainKernel(passwordNumber, numberOfPass, batchSize, &milliseconds,
            &h_passwords, &h_results, THREAD_PER_BLOCK, t);

    //TODO : save endingpoints on disk

    program_end = clock();
    program_time_used =
            ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);
}

__global__ void ntlm_chain_kernel2(Password * passwords, Digest * digests, int chainLength) {

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=0; i<chainLength; i++){
        ntlm(&passwords[index], &digests[index]);
        //TODO reduction(&passwords[index], &digests[index]);
    }
}

int main() {

    int passwordNumber = DEFAULT_PASSWORD_NUMBER;

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    return generateChains(passwords, result, passwordNumber, numberOfPass);
}