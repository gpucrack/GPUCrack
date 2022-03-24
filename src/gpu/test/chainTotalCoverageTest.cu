#include "chainTotalCoverageTest.cuh"

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

int main(int argc, char *argv[]){

    unsigned char charset[CHARSET_LENGTH] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                                             'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                                             'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                                             'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

    char *start_path;
    char *end_path;
    int pwd_length = atoi(argv[1]);

    long domain = pow(CHARSET_LENGTH, pwd_length);

    long idealM0 = (long)(0.6*(double)domain);

    long idealMtMax = (long)((double)((double)idealM0/(double)19.83));

    long mtMax = getNumberPassword(atoi(argv[2]), pwd_length);

    mtMax = idealMtMax;

    long passwordNumber = idealM0;
    //long passwordNumber = 18980;

    int t = computeT(mtMax, pwd_length);
    //int t = 500;

    //mtMax = 949;

    printf("mtMax: %ld\n", mtMax);

    printf("m0: %ld\n", passwordNumber);

    printf("Password length: %d\n", pwd_length);
    printf("Number of columns (t): %d\n\n", t);

    Password * passwords;

    printf("Password to be stored in ram: %ld\n", passwordNumber*t);

    initPasswordArray(&passwords, passwordNumber*t);

    // Adjust t depending on the chain length you want to test

    for(int i=0; i<t;i++) {
            for (long j = 0; j < passwordNumber; j++) {
                // Generate one password
                long counter = j;
                for (int q=0; q<pwd_length; q++) {
                    passwords[(i*passwordNumber)+j].bytes[q] = charset[counter % CHARSET_LENGTH];
                    counter /= CHARSET_LENGTH;
                }
            }

        printf("Before starting: ");
        printPassword(&(passwords[i*passwordNumber]));
        printf("\n");
        printf("%d: ", i);
        generateChains(&(passwords[i*passwordNumber]), passwordNumber, 1, i,
                       false, THREAD_PER_BLOCK, false, false, NULL, pwd_length, start_path, end_path);
        printPassword(&(passwords[i*passwordNumber]));
        printf("\n");
    }

    printf("Generation done\n");

    int nbFound = 0;
    int nbNotFound = 0;
    int nbDuplicates = 0;

    Password * result = (Password *)malloc(pwd_length);

    for(int i=0; i<5000; i++){
        // Generate one password
        long counter = i;
        for (int b = 0; b<pwd_length; b++) {
            (*result).bytes[b] = charset[counter % CHARSET_LENGTH];
            counter /= CHARSET_LENGTH;
        }
        bool found = false;
        for(int k=0; k < passwordNumber*t; k++){
            if((memcmp(&(*result).bytes, &(passwords[k].bytes), pwd_length) == 0) && !found){
                nbFound++;
                found = true;
                /*
                printf("Trouvé! ");
                for(int q=0; q<pwd_length; q++){
                    printf("%c", (*result).bytes[q]);
                }
                printf("\n");
                 */
            }else if (memcmp(&(*result).bytes, &(passwords[k].bytes), pwd_length) == 0 && found){
                nbDuplicates++;
            }else if ((k == (passwordNumber*t)-1) && !found){
                nbNotFound++;
                /*
                printf("Pas trouvé!! ");
                for(int q=0; q<pwd_length; q++){
                    printf("%c", (*result).bytes[q]);
                }
                printf("\n");
                */
            }
        }
        if ((i % 1000) == 0) printf("%d \n", i);
    }
    printPassword(&passwords[0]);
    printf("\n");
    printPassword(&passwords[(passwordNumber*t)-1]);
    printf("\n");
    printf("Number of passwords found: %d\n", nbFound);
    printf("Number of passwords not found: %d\n", nbNotFound);
    printf("Coverage: %f\n", ((double)(double)nbFound / (double)domain) * 100);
    printf("Number of duplicates: %d\n", nbDuplicates);
    printf("Percentage of duplicates: %f\n", ((double)(double)nbDuplicates / (double)(passwordNumber*t)) * 100);

    cudaFreeHost(passwords);

    return 0;
}
