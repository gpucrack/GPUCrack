#include "chainsV2.cuh"

__device__ static const unsigned char charset[64] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                                                      'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                                                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                                                      'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                                                      's', 't',
                                                      'u', 'v', 'w', 'x', 'y', 'z', '-', '_'};

__host__ void generateChains(Password * h_passwords, Digest * h_results, int passwordNumber, int numberOfPass) {

    printf("\n==========INPUTS==========\n");
    for(int i=passwordNumber-1; i<passwordNumber; i++) {
        printPassword(&(h_passwords[i]));
        printDigest(&(h_results[i]));
    }
    printf("\n");

    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    float milliseconds = 0;

    int batchSize = computeBatchSize(numberOfPass, passwordNumber);

    //TODO : compute t
    int t = 5;

    chainKernel(passwordNumber, numberOfPass, batchSize, &milliseconds,
            &h_passwords, &h_results, THREAD_PER_BLOCK, t);

    //TODO : save endingpoints on disk

    program_end = clock();
    program_time_used =
            ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);

    printf("\n==========OUTPUTS==========\n");
    for(int i=passwordNumber-1; i<passwordNumber; i++) {
        printPassword(&(h_passwords[i]));
        printDigest(&(h_results[i]));
    }
    printf("\n");

}

__device__ void reduce_digest(unsigned int index, Digest * digest, Password  * plain_text) {
    if (index == 666){
        printf("INPUT\n");
        printPassword(plain_text);
    }
    for (int i = 0; i < PASSWORD_LENGTH - 1; i++) {
        (*plain_text).bytes[i] = charset[((*digest).bytes[i] + index) % 64];
    }
    if (index == 666){
        printf("OUTPUT\n");
        printPassword(plain_text);
    }

}

__global__ void ntlm_chain_kernel2(Password * passwords, Digest * digests, int chainLength) {

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=0; i<chainLength; i++){
        ntlm(&passwords[index], &digests[index]);
        reduce_digest(index ,&digests[index], &passwords[index]);
    }
}

int main() {

    int passwordNumber = DEFAULT_PASSWORD_NUMBER;

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    generateChains(passwords, result, passwordNumber, numberOfPass);

    return 0;
}