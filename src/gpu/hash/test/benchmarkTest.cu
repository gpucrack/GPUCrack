#include "benchmarkTest.cuh"

void benchmark(int passwordNumber) {
    double maxHashRate = 0;
    double bestHashRateMean = 0;
    int bestValue = 0;

    Password * passwords;
    Digest * result;

    initArrays(&passwords, &result, passwordNumber);

    auto numberOfPass = memoryAnalysis(passwordNumber);

    for (int k = 2; k <= MAX_THREAD_NUMBER; k = k * 2) {

        double hashRateSum = 0;

        for (int i = 0; i < NUMBER_OF_TEST; i++) {

            float milliseconds = 0;

            hashTime(passwords, result, passwordNumber, &milliseconds, k,
                     numberOfPass);

            double hashrate = (passwordNumber / (milliseconds / 1000)) / 1000000;

            if (hashrate > maxHashRate) {
                maxHashRate = hashrate;
            }

            hashRateSum += hashrate;
        }

        double currentHashRateMean = hashRateSum / NUMBER_OF_TEST;
        if (currentHashRateMean > bestHashRateMean) {
            bestValue = k;
            bestHashRateMean = currentHashRateMean;
        }
    }



    printf("MAX HASHRATE : %f\n", maxHashRate);
    printf("BEST THREAD PER BLOCK VALUE : %d WITH MEAN : %f\n", bestValue, bestHashRateMean);

    cudaFreeHost(passwords);
    cudaFreeHost(result);
}

int main() {

    benchmark(DEFAULT_PASSWORD_NUMBER);

    return 0;
}


