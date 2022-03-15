#include "chainCoverageTest.cuh"

int cmpstr(Password *v1, Password *v2, int len) {
    return memcmp(v1, v2, len);
}

void swap(Password *v1, Password *v2, long size) {
    Password * buffer = (Password*) malloc(size);

    memcpy(buffer, v1, size);
    memcpy(v1, v2, size);
    memcpy(v2, buffer, size);

    free(buffer);
}

void q_sort(Password *v, long size, long left, long right) {
    Password *vt, *v3;
    long i, last, mid = (left + right) / 2;
    if (left >= right) {
        return;
    }

    // v left value
    Password *vl = (Password *) (v + (left));
    // v right value
    Password *vr = (Password *) (v + (mid));

    swap(vl, vr, size);

    last = left;
    for (i = left + 1; i <= right; i++) {

        // vl and vt will have the starting address
        // of the elements which will be passed to
        // comp function.
        vt = (Password *) (v + i);
        if (cmpstr(vl, vt, size) > 0) {
            ++last;
            v3 = (Password *) (v + (last));
            swap(vt, v3, size);
        }
    }
    v3 = (Password *) (v + (last));
    swap(vl, v3, size);
    q_sort(v, size, left, last - 1);
    q_sort(v, size, last + 1, right);
}


long dedup(Password *v, int size, long mt) {
    long index = 1;
    for (long i = 1; i < mt; i++) {
        Password *prev = (Password *) (v + ((i - 1)));
        Password *actual = (Password *) (v + (i));

        if (cmpstr(prev, actual, size) != 0) {
            Password *indexed = (Password *) (v + (index));
            memcpy(indexed, actual, size);
            index++;
        }
    }
    return index;
}

int main(){

    int pwd_length = 2;

    getNumberPassword(1, pwd_length);

    long domain = pow(CHARSET_LENGTH, pwd_length);

    printf("Domain: %ld\n", domain);

    long passwordNumber = (long)(0.1*(double)domain);

    printf("m0: %ld\n", passwordNumber);

    long mtMax = (long)((double)passwordNumber/19.83);

    Password * passwords;

    int t = computeT(mtMax, pwd_length);

    printf("Password to be stored in ram: %ld\n", passwordNumber*t);

    printf("Number of columns (t): %d\n\n", t);

    printf("mtMax: %ld\n", mtMax);

    initPasswordArray(&passwords, passwordNumber*t);

    char * start_path = (char *) "testStart.bin";
    char * end_path = (char *) "testEnd.bin";

    // Adjust t depending on the chain length you want to test

    for(int i=0; i<t;i++) {
        printf("%d: ", i);
        generateChains(&(passwords[i*passwordNumber]), passwordNumber, 1, i,
                       false, THREAD_PER_BLOCK, false, false, NULL, pwd_length, start_path, end_path);
        printPassword(&(passwords[i*passwordNumber]));
        printf("\n");
    }

    printf("Generation done\n");

    q_sort(passwords, sizeof(char)*pwd_length, 0, (passwordNumber*t) - 1);

    printf("Sorting done\n");

    long newlen = dedup(passwords, sizeof(char)*pwd_length, (passwordNumber*t));

    printf("Dedup done: %d\n", newlen);

    for(int n=0; n<newlen; n++){
        printPassword(&passwords[n]);
        printf("\n");
    }

    unsigned char charset[CHARSET_LENGTH] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                                             'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                                             'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                                             'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    char charsetLength = 61;

    int nbFound = 0;

    Password * result = (Password *)malloc(pwd_length);

    for(int i=0; i<domain; i++){
        // Generate one password
        long counter = i;
        for (int b = 0; b<pwd_length; b++) {
            (*result).bytes[b] = charset[counter % CHARSET_LENGTH];
            counter /= CHARSET_LENGTH;
        }

        for(int k=0; k < newlen; k++){
            if(memcmp(&(*result).bytes, &(passwords[k].bytes), pwd_length) == 0){
                nbFound++;
                printf("Trouvé! ");
                for(int q=0; q<pwd_length; q++){
                    printf("%c", (*result).bytes[q]);
                }
                printf("\n");
                break;
            }else if (k == newlen-1){
                printf("Pas trouvé!! ");
                for(int q=0; q<pwd_length; q++){
                    printf("%c", (*result).bytes[q]);
                }
                printf("\n");
            }
        }
        if ((i % 1000) == 0) printf("%d \n", i);
    }
    printPassword(&passwords[0]);
    printf("\n");
    printPassword(&passwords[newlen-1]);
    printf("\n");
    printf("Number of passwords found: %d\n", nbFound);
    printf("Coverage: %f\n", ((double)(double)nbFound / (double)domain) * 100);

    cudaFreeHost(passwords);

    return 0;
}
