#include "generalTest.cuh"

__host__ Password * generatePasswords() {

    auto * result = (Password*) malloc(PASSWORD_NUMBER*sizeof(Password));
    int random;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 9); // define the range

    // Generate all passwords
    for(int j=0; j<PASSWORD_NUMBER; j++) {
        Password currentPassword = *((Password*) malloc(sizeof(Password)));
        // Generate one password
        for (unsigned char & byte : currentPassword.bytes) {
            random = distr(gen);
            byte = random;
        }

        // Debug
        //for (unsigned char byte : currentPassword.bytes) {
        //    printf("%x", byte);
        //}
        //printf("\n");

        result[j] = currentPassword;
    }

    return result;
}

int main() {

    Password * passwords = generatePasswords();

    return(parallelized_hash(passwords));

}