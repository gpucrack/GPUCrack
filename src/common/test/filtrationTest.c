#include "filtrationTest.h"

int main(int argc, char *argv[]) {

    char * path = (char *) "test";

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

    // Clean the table by deleting duplicate endpoints
    long *res = filter(startName, endName, startNameF, endNameF, 1, 56800237, NULL);
    if (res[2] == res[3]) {
        printf("Filtration done!\n\n");
        printf("The files have been generated with success.\n");
    }
}