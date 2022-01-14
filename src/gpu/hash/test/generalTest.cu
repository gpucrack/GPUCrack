#include "generalTest.cuh"

__host__ Password * generatePasswords() {

    auto * result = (Password*) malloc(PASSWORD_NUMBER*sizeof(Password));
    int random;
    // Generate all passwords
    for(int j=0; j<PASSWORD_NUMBER; j++) {
        Password currentPassword = *((Password*) malloc(sizeof(Password)));

        // Generate one password
        for (int i = 0; i < PASSWORD_LENGTH; i++) {
            srand((unsigned) time(nullptr));
            random = (rand() % 9);
            currentPassword.bytes[i] = random;
        }

        // Debug
        for(int n=0; n<PASSWORD_LENGTH; n++) {
            printf("%x", currentPassword.bytes[n]);
            printf("\n");
        }

        result[j] = currentPassword;
    }

    return result;
}

int main() {
    generatePasswords();
}