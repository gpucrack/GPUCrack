/*
 * Author: Maxime Missichini
 * Email: missichini.maxime@gmail.com
 * -----
 * File: parallelized_hash.cu
 * Created Date: 28/09/2021
 * Last Modified: 08/11/2021
 * -----
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "myMd5.cu"
#include <ctime>
#include "test_hash.cuh"

#define NUMBER_OF_PASSWORD 1000000
#define MAX_PASSWORD_LENGTH 15

int main(){

    double program_time_used;
    clock_t program_start, program_end;
    program_start = clock();

    //Host copies
    BYTE ** passwords_to_hash;
    BYTE ** h_results;
    BYTE ** file_buffer;
    WORD total_length = NUMBER_OF_PASSWORD;
    WORD length = MAX_PASSWORD_LENGTH;

    //We store everything inside arrays of pointers to char pointers into host memory first
    passwords_to_hash = (BYTE**)malloc(sizeof(BYTE*)*NUMBER_OF_PASSWORD);
    h_results = (BYTE**)malloc(sizeof(BYTE*)*NUMBER_OF_PASSWORD);
    file_buffer = (BYTE**)malloc(sizeof(BYTE*)*(NUMBER_OF_PASSWORD));

    for(int i=0;i<NUMBER_OF_PASSWORD;i++){
        file_buffer[i] = (BYTE*)malloc(sizeof(BYTE)*MAX_PASSWORD_LENGTH);
    }

    //Opening the file with passwords to hash
    FILE * fp = fopen("/home/mynder/Desktop/projets/GPUCrack/src/gpu/passwords.txt","r");

    if(fp==nullptr){
        perror("Error while opening the file\n");
        exit(EXIT_FAILURE);
    }

    int n = 0;
    while(n<NUMBER_OF_PASSWORD){
        fgets((char*)file_buffer[n],MAX_PASSWORD_LENGTH,fp);
        n++;
    }

    printf("PASSWORD FILE TO BUFFER DONE @ %f seconds\n",(double)(clock()-program_start)/CLOCKS_PER_SEC);

    //Deep copy, this is a mecanism for CUDA to allocate memory correctly
    for(int i=0;i<NUMBER_OF_PASSWORD;i++){
        //Each time we allocate the host pointer into device memory
        cudaMalloc((void**)&passwords_to_hash[i], MAX_PASSWORD_LENGTH*sizeof(BYTE));
        cudaMalloc((void**)&h_results[i], MAX_PASSWORD_LENGTH*sizeof(BYTE));
        //We also copy the passwords to the device memory
        cudaMemcpy(passwords_to_hash[i],file_buffer[i],MAX_PASSWORD_LENGTH*sizeof(BYTE),cudaMemcpyHostToDevice);
    }

    printf("COPY TO GPU DONE @ %f seconds\n",(double)(clock()-program_start)/CLOCKS_PER_SEC);

    fclose(fp);

    //Device copies
    BYTE **d_passwords;
    BYTE **d_results;
    WORD *d_total_length;
    WORD *d_length;
    cudaMalloc((void**)&d_passwords,sizeof(BYTE*)*NUMBER_OF_PASSWORD);
    cudaMalloc((void**)&d_results,sizeof(BYTE*)*NUMBER_OF_PASSWORD);
    cudaMalloc((void**)&d_total_length,sizeof(WORD*));
    cudaMalloc((void**)&d_length,sizeof(WORD*));
    cudaMemcpy(d_passwords,passwords_to_hash,sizeof(BYTE*)*NUMBER_OF_PASSWORD,cudaMemcpyHostToDevice);
    cudaMemcpy(d_results,h_results,sizeof(BYTE*)*NUMBER_OF_PASSWORD,cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_length,&total_length,sizeof(WORD),cudaMemcpyHostToDevice);
    cudaMemcpy(d_length,&length,sizeof(WORD),cudaMemcpyHostToDevice);

    printf("CREATE VARIABLES DONE @ %f seconds\n",(double)(clock()-program_start)/CLOCKS_PER_SEC);

    //Mesure time
    clock_t start, end;
    double gpu_time_used;
    start = clock();

    //We need to define the context for MD5 hash
    CUDA_MD5_CTX context;
    kernel_md5_hash<<<NUMBER_OF_PASSWORD-(NUMBER_OF_PASSWORD/512),NUMBER_OF_PASSWORD/512>>>(d_passwords,d_total_length,d_results,d_length,context);

    printf("KERNEL DONE @ %f seconds\n",(double)(clock()-program_start)/CLOCKS_PER_SEC);

    //Check for errors during kernel execution
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        return 1;
    }

    end = clock();
    gpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;

    //Then, in order to copy back we need a host array of pointers to char pointers
    BYTE ** results;
    results = (BYTE **)malloc(NUMBER_OF_PASSWORD*sizeof(BYTE*));

    //We need to allocate each char pointers
    for(int k=0;k<NUMBER_OF_PASSWORD;k++){
        results[k] = (BYTE*)malloc(MAX_PASSWORD_LENGTH*sizeof(BYTE));
    }

    printf("CREATE FINAL RESULT ARRAY DONE @ %f seconds\n",(double)(clock()-program_start)/CLOCKS_PER_SEC);

    //Copy back the device result array to host result array
    cudaMemcpy(h_results,d_results,sizeof(BYTE*)*NUMBER_OF_PASSWORD,cudaMemcpyDeviceToHost);

    printf("RETRIEVING RESULTS ...\n");
    int j;
    //Deep copy of each pointers to the host result array
    for(j=0;j<NUMBER_OF_PASSWORD;j++){
        cudaMemcpy(results[j],h_results[j],MAX_PASSWORD_LENGTH*sizeof(BYTE),cudaMemcpyDeviceToHost);
    }
    printf("PASSWORD RETRIEVED : %d\n",j);

    printf("GPU PARALLEL HASH TIME : %f seconds\n",gpu_time_used);

    //Cleanup
    free(passwords_to_hash);
    free(h_results);
    free(results);
    free(file_buffer);
    cudaFree(d_passwords);
    cudaFree(d_results);
    cudaFree(passwords_to_hash);
    cudaFree(h_results);

    program_end = clock();
    program_time_used = ((double)(program_end - program_start))/CLOCKS_PER_SEC;
    printf("TOTAL EXECUTION TIME : %f seconds\n",program_time_used);

    return 0;
}
