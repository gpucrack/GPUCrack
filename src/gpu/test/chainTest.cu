#include "chainTest.cuh"

int main(){
    int passwordNumber = getNumberPassword(1, 5);

    Password * passwords;
    Digest * result;

    // check
    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysisGPU(passwordNumber);

    char * start_path = (char *) "testStart.bin";
    char * end_path = (char *) "testEnd.bin";

    // Adjust t depending on the chain length you want to test
    generateChains(passwords, passwordNumber, numberOfPass, 3964,
                   false, THREAD_PER_BLOCK, true, true, result, 5, start_path, end_path);

    printf("Should be first password inside endpoints:\n");
    printPassword(&passwords[0]);
    printf("\n");


    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}
