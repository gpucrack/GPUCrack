#include "reduction.cuh"

//char *charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";   // the characters to be used in the generated plain text

static void HandleError(	cudaError_t err,
                            const char *file,
                            int line )
{
    if (err != cudaSuccess)
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/*
 * Reduces a hash into a plain text of length PLAIN_LENGTH.
 * index: index of the column in the table
 * hash: cypher text to be reduced
 * plain: result of the reduction

void reduce(unsigned long int index, const char *hash, char *plain) {
    for (unsigned long int i = 0; i < PLAIN_LENGTH; i++, plain++, hash++)
        *plain = charset[(unsigned char) (*hash ^ index) % CHARSET_LENGTH];
}*/
__device__ char char_in_range(unsigned char n) {
    const char *chars =
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";

    return chars[n];
}

__device__ void create_startpoint(unsigned long counter, Password *plain_text) {
    for (int i = PASSWORD_LENGTH - 1; i >= 0; i--) {
        plain_text->bytes[i] = char_in_range(counter % 64);
        counter /= 64;
    }
}

__global__ void reduce_digest(unsigned long iteration, Digest **digests, Password **plain_texts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // pseudo-random counter based on the hash
    unsigned long counter = digests[idx]->bytes[7];
    for (char i = 6; i >= 0; i--) {
        counter <<= 8;
        counter |= digests[idx]->bytes[i];
    }
    create_startpoint(counter + iteration, plain_texts[idx]);
}


/*
__global__ void reduce_kernel(int index, Digest * hashes, Password * plains) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Est-ce que cette vérification est nécéssaire ?
    if (idx < DEFAULT_PASSWORD_NUMBER)
    {
        for (int i = 0; i < PASSWORD_LENGTH; i++, plains[idx]++, hashes[idx]++)
            *plains[idx] = charset[(unsigned char) (*hashes[idx] ^ index) % CHARSET_LENGTH];
    }
}
*/
/*
 * Tests the reduction and displays it in the console.
 */
int main() {;
    int N = DEFAULT_PASSWORD_NUMBER;

    // Tableau CPU
    Digest * pHashes = NULL;
    Password * pPasswords = NULL;

    // Tableau GPU : à ce stade, une adresse mémoire
    Digest * dHashes = NULL;
    Password * dPasswords = NULL;

    //  Allocation de la mémoire CPU
    pHashes	= (Digest*)malloc(sizeof(Digest) * N);
    pPasswords = (Password*)malloc(sizeof(Password) * N);

    // Allocation de la mémoire sur le GPU
    HANDLE_ERROR(cudaMalloc((void**)&dHashes,  sizeof(Digest) * N));
    HANDLE_ERROR(cudaMalloc((void**)&dPasswords, sizeof(Password) * N));

    // Initialisation des hashes
    for(int i = 0; i < N; i++)
    {
        for (unsigned char &byte: pHashes[i].bytes) {
            byte = 'h';
        }
        for (unsigned char &byte: pPasswords[i].bytes) {
            byte = 'p';
        }
    }

    // Copie des tableaux sur le GPU
    HANDLE_ERROR(cudaMemcpy(dHashes, pHashes, sizeof(Digest) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dPasswords, pPasswords, sizeof(Password) * N, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);
    reduce_digest<<<grid_size, block_size>>>(1, &dHashes, &dPasswords);

    // Copie des données du GPU sur le CPU
    pPasswords = (Password*)malloc(sizeof(Password) * N);
    //HANDLE_ERROR(cudaMemcpy(pPasswords, dPasswords, sizeof(Password) * N, cudaMemcpyDeviceToHost));

    for(int i=0; i<N; i++) {
        // On affiche la première sortie du résultat :
        for (unsigned char byte: pPasswords[i].bytes) {
            printf("%x", byte);
        }
        printf("\n");
    }

    // Libération de la mémoire
    free(pHashes);
    free(pPasswords);
    //HANDLE_ERROR(cudaFree(dHashes));
    //HANDLE_ERROR(cudaFree(dPasswords));

    return(0);
}