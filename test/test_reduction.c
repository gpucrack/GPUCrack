#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../src/common/reduction.h"

double test_function1(long int nb_iter) {
    clock_t start, end;
    double cpu_time_used;
    char* hash = "8846f7eaee8fb117ad06bdd830b7586c";

    char* plain = malloc(sizeof(char) * PLAIN_LENGTH);

    printf("Starting reduction...\n");

    start = clock();
    for (unsigned long int i = 0; i < nb_iter; i++) {
        reduceV1(i, hash, plain);
    }
    end = clock();
    printf("Completed.\n");
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time : %fs\n\n", cpu_time_used);
    return cpu_time_used;
}

double test_function2(long int nb_iter) {
    clock_t start, end;
    double cpu_time_used;
    char* hash = "8846f7eaee8fb117ad06bdd830b7586c";

    char* plain = malloc(sizeof(char) * PLAIN_LENGTH);

    printf("Starting reduction...\n");

    start = clock();
    for (unsigned long int i = 0; i < nb_iter; i++) {
        reduceV2(i, hash, plain);
    }
    end = clock();
    printf("Completed.\n");
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time : %fs\n\n", cpu_time_used);
    return cpu_time_used;
}

double test_function3(long int nb_iter) {
    clock_t start, end;
    double cpu_time_used;
    char* hash = "8846f7eaee8fb117ad06bdd830b7586c";

    char* plain = malloc(sizeof(char) * PLAIN_LENGTH);

    printf("Starting reduction...\n");

    start = clock();
    for (unsigned long int i = 0; i < nb_iter; i++) {
        reduceV3(i, hash, plain);
    }
    end = clock();
    printf("Completed.\n");
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time : %fs\n\n", cpu_time_used);
    return cpu_time_used;
}

double test_function4(long int nb_iter) {
    clock_t start, end;
    double cpu_time_used;
    char* hash = "8846f7eaee8fb117ad06bdd830b7586c";

    char* plain = malloc(sizeof(char) * PLAIN_LENGTH);

    printf("Starting reduction...\n");

    start = clock();
    for (unsigned long int i = 0; i < nb_iter; i++) {
        reduceV4(i, hash, plain);
    }
    end = clock();
    printf("Completed.\n");
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time : %fs\n\n", cpu_time_used);
    return cpu_time_used;
}

double test_function5(long int nb_iter) {
    clock_t start, end;
    double cpu_time_used;
    char* hash = "8846f7eaee8fb117ad06bdd830b7586c";

    char* plain = malloc(sizeof(char) * PLAIN_LENGTH);

    printf("Starting reduction...\n");

    start = clock();
    for (unsigned long int i = 0; i < nb_iter; i++) {
        reduceV5(i, hash, plain);
    }
    end = clock();
    printf("Completed.\n");
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time : %fs\n\n", cpu_time_used);
    return cpu_time_used;
}


int main(int argc, char** argv) {
    long int nb_reductions = 1000000000;    // 100 millions
    int nb_iter = 5;
    double sum, avg = 0;

    double times1[nb_iter];
    
    for (int i = 0; i<nb_iter; i++) {
        printf("reducev5\n");
        times1[i] = test_function5(nb_reductions);
        sum += times1[i];
    }

    printf("Average for %ld reductions with reducev5 is %f seconds (%d iterations). \n Plain text length is %d. Charset contains %d characters.\n", nb_reductions, sum/nb_iter, nb_iter, PLAIN_LENGTH, CHARSET_LENGTH);
}