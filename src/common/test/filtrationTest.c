#include "filtrationTest.h"

// C implementation of Heap Sort
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

    char * path = (char *) "table6";

    char startName[100];
    char endName[100];
    strcpy(startName, path);
    strcat(startName, "_start_");
    strcpy(endName, path);
    strcat(endName, "_end_");
    char tableChar[10];
    sprintf(tableChar, "%d", 0);
    strcat(startName, tableChar);
    strcat(startName, ".bin");
    strcat(endName, tableChar);
    strcat(endName, ".bin");

    char startNameF[100];
    char endNameF[100];
    strcpy(startNameF, path);
    strcat(startNameF, "_start_");
    strcpy(endNameF, path);
    strcat(endNameF, "_end_");
    sprintf(tableChar, "%d", 0);
    strcat(startNameF, tableChar);
    strcat(startNameF, "f.bin");
    strcat(endNameF, tableChar);
    strcat(endNameF, "f.bin");

    printf("Engaging filtration...\n");

    // Start the timer
    clock_t start = clock();

    int blocks = 64;

    // Clean the table by deleting duplicate endpoints
    long *res = filter(startName, endName, startNameF, endNameF, 1 * blocks, 2840011780 / blocks, "table6", 0, NULL);

    // Stop the timer
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    if (res[2] == res[3]) {
        printf("Filtration done in %f sec.\n", time_spent);
    }
}
