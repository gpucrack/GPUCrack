#include "generalTest.cuh"

int main() {

    int passwordNumber = 134217728;

    Password * passwords = generatePasswords(passwordNumber);

    auto * result = parallelized_hash(passwords, passwordNumber);

    free(passwords);
    free(result);

    return(0);
}