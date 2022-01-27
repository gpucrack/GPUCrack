#include "chainsV2.cuh"

__host__ int createChain() {
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

int main() {
    return createChain();
}