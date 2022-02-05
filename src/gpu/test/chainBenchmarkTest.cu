#include "chainBenchmarkTest.cuh"

int main(){
    int passwordNumber = getNumberPassword(8);

    double maxHashRedRate = 0;
    double bestHashRedRateMean = 0;
    int bestValue = 0;

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    // int numberOfColumn = computeT(16);
    int numberOfColumn = 100;

    float milliseconds = 0;
    float tempMilli = 0;

    for (int k = 2; k <= MAX_THREAD_NUMBER; k = k * 2) {

        double hashRedRateSum = 0;

        for (int i = 0; i < NUMBER_OF_TEST; i++) {

            // Measure GPU time
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);

            cudaEventRecord(start);
            generateChains(passwords, result, passwordNumber,
                           numberOfPass, numberOfColumn, false, k);
            cudaEventRecord(end);
            cudaEventSynchronize(end);

            cudaEventElapsedTime(&tempMilli, start, end);
            milliseconds += tempMilli;
            cudaEventDestroy(start);
            cudaEventDestroy(end);

            double hashRedRate = (((float) (passwordNumber) / (tempMilli / 1000)) / 1000000) * (float) numberOfColumn;

            if (hashRedRate > maxHashRedRate) {
                maxHashRedRate = hashRedRate;
            }

            hashRedRateSum += hashRedRate;

        }

        double currentHashRateMean = hashRedRateSum / NUMBER_OF_TEST;
        if (currentHashRateMean > bestHashRedRateMean) {
            bestValue = k;
            bestHashRedRateMean = currentHashRateMean;
        }

    }

    printf("MAX HASHREDRATE : %f\n", maxHashRedRate);
    printf("BEST THREAD PER BLOCK VALUE : %d WITH MEAN : %f\n", bestValue, bestHashRedRateMean);
    printf("NOTE: THIS HASHREDRATE IS LOWER THAN THE REAL ONE\n");

    cudaFreeHost(passwords);
    cudaFreeHost(result);

    return 0;
}