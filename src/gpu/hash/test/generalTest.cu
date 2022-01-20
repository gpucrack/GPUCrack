#include "generalTest.cuh"

int main() {

    int passwordNumber = 268435456;

    // Simulate when we send password as input
    Password *passwords = generatePasswords(passwordNumber);

    auto *result = parallelized_hash(passwords, passwordNumber);

    free(passwords);
    free(result);

    passwordNumber = DEFAULT_PASSWORD_NUMBER;

    // Simulate when we send password as input
    passwords = generatePasswords(passwordNumber);

    result = parallelized_hash(passwords, passwordNumber);

    free(passwords);
    free(result);

    return (0);
}