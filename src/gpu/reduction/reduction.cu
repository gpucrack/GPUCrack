#include "reduction.cuh"

// char *charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";   // the characters to be used in the generated plain text

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

__global__ void reduce_kernel(unsigned long int index, Digest * hashes, Password * plains) {
    printf("Hey");
}

/*
 * Tests the reduction and displays it in the console.
 */
int main() {;
    int N = 1000;

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
        pHashes[i] = "8846F7EAEE8FB117AD06BDD830B7586C";
        pPasswords[i] = "PASSWD";
    }

    // Copie des tableaux sur le GPU
    HANDLE_ERROR(cudaMemcpy(dHashes, pHashes, sizeof(Digest) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dPasswords, pPasswords, sizeof(Password) * N, cudaMemcpyHostToDevice));

    // int block_size = 256;
    // int grid_size = ((N + block_size) / block_size);
    reduce_kernel<<<1, 1>>>(1, dHashes, dPasswords);

    // Copie des données du GPU sur le CPU
    pPasswords = (Password*)malloc(sizeof(Password) * N);
    HANDLE_ERROR(cudaMemcpy(pPasswords, dPasswords, sizeof(Password) * N, cudaMemcpyDeviceToHost));

    // On affiche la première sortie du résultat :
    std::cout << "First value in CPU buffer : " << pPasswords[0][0] << std::endl;

    // Libération de la mémoire
    free(pHashes);
    free(pPasswords);
    HANDLE_ERROR(cudaFree(dHashes));
    HANDLE_ERROR(cudaFree(dPasswords));

    return(0);
}