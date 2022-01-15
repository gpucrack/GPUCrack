#include "generalTest.cuh"

__host__ Password * generatePasswords(long passwordNumber) {

    auto * result = (Password*) malloc(passwordNumber*sizeof(Password));

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 9); // define the range

    printf("\n==========GENERATING PASSWORDS==========\n");
    // Generate all passwords
    for(int j=0; j<passwordNumber; j++) {
        auto * currentPassword = (Password *) malloc(sizeof(Password));
        // Generate one password
        for (unsigned char &byte: (*currentPassword).bytes) {
            byte = distr(gen);
        }

        // Debug
        //for (unsigned char &byte : (*currentPassword).bytes) {
        //    printf("%x", byte);
        //}
        //printf("\n");

        result[j] = *(currentPassword);
        free(currentPassword);

        // Debug
        //for (unsigned char &byte : result[j].bytes) {
        //    printf("%x", byte);
        //}
        //printf("\n");

    }

    printf("====================\n");

    return result;
}

int main() {

    int passwordNumber = 536870912;

    Password * passwords = generatePasswords(passwordNumber);

    auto * result = parallelized_hash(passwords, passwordNumber);

    free(result);

    return(0);
}