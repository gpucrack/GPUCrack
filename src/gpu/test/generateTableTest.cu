#include "generateTableTest.cuh"

int main(int argc, char *argv[]) {

    if (argc < 4) {
        printf("Error: not enough arguments given.\n"
               "Usage: 'generateTable c n p (path)', where:\n"
               "     - c is the passwords' length (in characters).\n"
               "     - n is the number of tables to generate.\n"
               "     - p is the ratio of the password's length domain to be used as m0.\n"
               "     - (optional) path is the path to the start and end points files to create.\n");
        exit(1);
    }

    printSignature();

    unsigned long long parameters[5];

    bool debug = false;

    computeParameters(parameters, argc, argv, debug);

    Password * passwords;

    generateTables(parameters, passwords, argc, argv, debug);

    return 0;
}
