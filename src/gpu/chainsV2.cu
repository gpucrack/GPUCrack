#include "chainsV2.cuh"

__host__ int generateChains() {
    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    int t = 0;

    // Cleanup

    program_end = clock();
    program_time_used =
            ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n", program_time_used);
}

__global__ void ntlm_chain_kernel2(Password * d_passwords, Digest * d_results, int chainLength) {

}

int main() {
    return generateChains();
}