#include "generateTableTest.cuh"

int main(int argc, char *argv[]) {

    if (argc < 4) {
        printf("Error: not enough arguments given.\n"
               "Usage: 'generateTable c n mt (path)', where:\n"
               "     - c is the passwords' length (in characters).\n"
               "     - n is the number of tables to generate.\n"
               "     - mt is the number of end points to be generated.\n"
               "     - (optional) path is the path to the start and end points files to create.\n");
        exit(1);
    }

    printSignature();

    unsigned long long parameters[5];

    computeParameters(parameters, argc, argv, true);

    unsigned long long passwordNumber = parameters[0];
    unsigned long long t = parameters[2];

    Password * passwords;

    unsigned long long nbOp = t * passwordNumber;

    printf("Number of crypto op: %lld\n", nbOp);

    generateTables(parameters, passwords, argc, argv);

    return 0;
}
