#include "reduction.h"

// Global variables for GPU
__device__ password_length_gpu = PASSWORD_LENGTH;
__device__ charset_length_gpu = CHARSET_LENGTH;

// The character set used for passwords. We declare it in the host scope and in the device scope.
__device__ static const char *charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";
static const char *charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";

// The character set used for digests (NTLM hashes).
static const char hashset[16] = {0x88, 0x46, 0xF7, 0xEA, 0xEE, 0x8F, 0xB1, 0x17, 0xAD, 0x06, 0xBD, 0xD8, 0x30, 0xB7,
                                 0x58, 0x6C};

void display_password(Password &pwd, bool br = true) {
    for (unsigned char byte: pwd.bytes) {
        printf("%c", (char) byte);
    }
    if (br) printf("\n");
}

void display_passwords(Password **passwords) {
    for (int j = 0; j < DEFAULT_PASSWORD_NUMBER; j++) {
        display_password((*passwords)[j]);
    }
}

void display_digest(Digest &digest, bool br = true) {
    for (unsigned char byte: digest.bytes) {
        printf("%02X", byte);
    }
    if (br) printf("\n");
}

void display_digests(Digest **digests) {
    for (int j = 0; j < DEFAULT_PASSWORD_NUMBER; j++) {
        display_digest((*digests)[j]);
    }
}

void generate_digests_random(Digest **digests, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = HASH_LENGTH - 1; i >= 0; i--) {
            (*digests)[j].bytes[i] = hashset[rand() % CHARSET_LENGTH];
        }
    }
}

__device__ void reduce_digest(unsigned long index, Digest &digest, Password &plain_text) {
    for (int i = 0; i < PASSWORD_LENGTH - 1; i++) {
        plain_text.bytes[i] = charset[(digest.bytes[i] + index) % CHARSET_LENGTH];
    }
}

__global__ void reduce_digests(Digest **digests, Password **plain_texts) {
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    reduce_digest(idx, (*digests)[idx], (*plain_texts)[idx]);
}

inline int pwdcmp(Password &p1, Password &p2) {
    for (int i = 0; i < CEILING(PASSWORD_LENGTH, 4); i++) {
        if (p1.i[i] != p2.i[i]) {
            return false;
        }
    }
    return true;
}

int count_duplicates(Password **passwords, bool debug = false) {
    int count = 0;
    for (int i = 0; i < DEFAULT_PASSWORD_NUMBER; i++) {
        if (debug) printf("Searching for duplicate of password number %d...\n", i);
        for (int j = i + 1; j < DEFAULT_PASSWORD_NUMBER; j++) {
            // Increment count by 1 if duplicate found
            if (pwdcmp((*passwords)[i], (*passwords)[j])) {
                printf("Found a duplicate : ");
                display_password((*passwords)[i]);
                count++;
            }
        }
    }
    return count;
}

void display_reductions(Digest **digests, Password **passwords, int n = DEFAULT_PASSWORD_NUMBER) {
    for (int i = 0; i < n; i++) {
        display_digest((*digests)[i], false);
        printf(" --> ");
        display_password((*passwords)[i], false);
        printf("\n");
    }
}

/*
 * Tests the GPU reduction speed and searches for duplicates in reduced hashes.
 */
int main() {

    // Initialize and allocate memory for a password array
    Password *passwords = NULL;
    passwords = (Password *) malloc(sizeof(Password) * DEFAULT_PASSWORD_NUMBER);

    // Initialize and allocate memory for a digest array
    Digest *digests = NULL;
    digests = (Digest *) malloc(sizeof(Digest) * DEFAULT_PASSWORD_NUMBER);

    // Generate DEFAULT_PASSWORD_NUMBER digests
    printf("Generating digests...\n");
    generate_digests_random(&digests, DEFAULT_PASSWORD_NUMBER);
    //display_digests(&digests);
    printf("Digest generation done!\n");

    // Allocate pinned memory for the password array
    cudaError_t status = cudaMallocHost(passwords, DEFAULT_PASSWORD_NUMBER * sizeof(Password), cudaHostAllocDefault);
    if (status != cudaSuccess)
        printf("An error occurred when allocating pinned host memory.\n");

    // Set block and grim dimensions
    int block_size = 256;
    int grid_size = ((DEFAULT_PASSWORD_NUMBER + block_size) / block_size);

    // Start the chronometer...
    float t = 0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    // Reduce all those digests into passwords
    reduce_digests<<<grid_size, block_size>>>(&digests, &passwords);

    // Stop the chronometer!
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&t, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    double reduce_rate = (DEFAULT_PASSWORD_NUMBER / (t / 1000)) / 1000000);

    printf("Reduction of %d digests ended after %f seconds.\nReduction rate: %f MR/s.\n", DEFAULT_PASSWORD_NUMBER,
           (double) t, reduce_rate);

    display_reductions(&digests, &passwords, 5);

    // Free memory
    cudaFreeHost(passwords);
    cudaFreeHost(result);

    /*int dup = count_duplicates(&passwords);
    printf("Found %d duplicate(s) among the %d reduced passwords (%f percent).\n", dup, DEFAULT_PASSWORD_NUMBER,
           ((double) dup / DEFAULT_PASSWORD_NUMBER) * 100);*/

    //display_passwords(&passwords);
    return 0;
}