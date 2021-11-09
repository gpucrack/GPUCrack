//
// Created by mynder on 09/11/2021.
//

#include <cstdio>
#include <cstdlib>
#include "test_hash.cuh"
#include "myMd5.cu"

#define REFERENCE_SENTENCE1 "The quick brown fox jumps over the lazy dog"
#define REFERENCE_RESULT1 "9e107d9d372bb6826bd81d3542a419d6"
#define REFERENCE_SENTENCE2 "The quick brown fox jumps over the lazy dog."
#define REFERENCE_RESULT2 "e4d909c290d0fb1ca068ffaddf22cbd0"
#define MAX_PASSWORD_LENGTH 100
#define NUMBER_OF_PASSWORD 2

int main(){

    BYTE ** passwords = (BYTE**)malloc(NUMBER_OF_PASSWORD*sizeof(BYTE*));
    BYTE ** results = (BYTE**)malloc(NUMBER_OF_PASSWORD*sizeof(BYTE*));
    WORD total_length = NUMBER_OF_PASSWORD;
    WORD length = MAX_PASSWORD_LENGTH;

    for(int i=0;i<NUMBER_OF_PASSWORD;i++){
        //Each time we allocate the host pointer into device memory
        cudaMalloc((void**)&passwords[i], MAX_PASSWORD_LENGTH*sizeof(BYTE));
        cudaMalloc((void**)&results[i], MAX_PASSWORD_LENGTH*sizeof(BYTE));
    }

    int count;
    for(count=0;count<MAX_PASSWORD_LENGTH;count++){
        if(REFERENCE_SENTENCE1[count] == '\0'){
            printf("COUNT : %d\n",count);
            break;
        }
    }
    cudaMemcpy(passwords[0],REFERENCE_SENTENCE1,count*sizeof(BYTE),cudaMemcpyHostToDevice);

    for(count=0;count<MAX_PASSWORD_LENGTH;count++){
        if(REFERENCE_SENTENCE2[count] == '\0'){
            printf("COUNT : %d\n",count);
            break;
        }
    }
    cudaMemcpy(passwords[1],REFERENCE_SENTENCE2,count*sizeof(BYTE),cudaMemcpyHostToDevice);

    BYTE **d_passwords;
    BYTE **d_results;
    WORD *d_total_length;
    WORD *d_length;

    cudaMalloc((void**)&d_passwords,sizeof(BYTE*)*NUMBER_OF_PASSWORD);
    cudaMalloc((void**)&d_results,sizeof(BYTE*)*NUMBER_OF_PASSWORD);
    cudaMalloc((void**)&d_total_length,sizeof(WORD*));
    cudaMalloc((void**)&d_length,sizeof(WORD*));
    cudaMemcpy(d_passwords,passwords,sizeof(BYTE*)*NUMBER_OF_PASSWORD,cudaMemcpyHostToDevice);
    cudaMemcpy(d_results,results,sizeof(BYTE*)*NUMBER_OF_PASSWORD,cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_length,&total_length,sizeof(WORD),cudaMemcpyHostToDevice);
    cudaMemcpy(d_length,&length,sizeof(WORD),cudaMemcpyHostToDevice);

    CUDA_MD5_CTX context;
    kernel_md5_hash<<<NUMBER_OF_PASSWORD,1>>>(d_passwords,d_total_length,d_results,d_length,context);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        return 1;
    }

    //Then, in order to copy back we need a host array of pointers to char pointers
    BYTE ** final_results;
    final_results = (BYTE **)malloc(NUMBER_OF_PASSWORD*sizeof(BYTE*));

    //We need to allocate each char pointers
    for(int k=0;k<NUMBER_OF_PASSWORD;k++){
        final_results[k] = (BYTE*)malloc(MAX_PASSWORD_LENGTH*sizeof(BYTE));
    }

    //Copy back the device result array to host result array
    cudaMemcpy(results,d_results,sizeof(BYTE*)*NUMBER_OF_PASSWORD,cudaMemcpyDeviceToHost);

    int j;
    //Deep copy of each pointers to the host result array
    for(j=0;j<NUMBER_OF_PASSWORD;j++){
        cudaMemcpy(final_results[j],results[j],MAX_PASSWORD_LENGTH*sizeof(BYTE),cudaMemcpyDeviceToHost);
    }
    printf("PASSWORD RETRIEVED : %d\n",j);

    bool test1 = true;
    bool test2 = true;

    for(int j=0; j<NUMBER_OF_PASSWORD;j++) {
        for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
                if (j==0) {
                    if (final_results[j][i] != REFERENCE_RESULT1[i]) {
                        test1 = false;
                        break;
                    }
                }else{
                    if (final_results[j][i] != REFERENCE_RESULT2[i]) {
                        test2 = false;
                        break;
                    }
                }
        }
    }

    printf("RESULTS :");
    for(int i=0;i<MAX_PASSWORD_LENGTH;i++){
        if(final_results[0][i] == '\0') break;
        printf("%x",final_results[0][i]);
    }
    printf(" ,");
    for(int i=0;i<MAX_PASSWORD_LENGTH;i++){
        if(final_results[0][i] == '\0') break;
        printf("%x",final_results[1][i]);
    }
    printf("\n");
    printf("TEST RESULTS : %d ,%d\n",test1,test2);

    //Cleanup
    free(passwords);
    free(results);
    free(final_results);
    cudaFree(d_passwords);
    cudaFree(d_results);
    cudaFree(passwords);
    cudaFree(results);

    return 0;

}