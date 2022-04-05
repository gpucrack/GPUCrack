#include "chainCoverageTest.cuh"

int main(int argc, char *argv[]){

    unsigned char charset[CHARSET_LENGTH] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                                             'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                                             'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                                             'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

    int pwd_length = PASSWORD_LENGTH;

    char * start_path = (char *) "testStart.bin";
    char * end_path = (char *) "testEnd.bin";

    FILE *fp;
    fp = fopen(start_path, "rb");

    if (fp == NULL)(exit(1));

    char buff[255];

    // Retrieve the start points file info
    fscanf(fp, "%s", buff);
    unsigned long mt;
    sscanf(buff, "%ld", &mt);
    printf("Number of points (mt): %lu\n", mt);
    fgets(buff, 255, (FILE *) fp);
    fgets(buff, 255, (FILE *) fp); // skip the pwd_length line

    int passwordNumber = (int)mt;

    // Retrieve the chain length (t)
    int t;
    fgets(buff, 255, (FILE *) fp);
    sscanf(buff, "%d", &t);
    printf("Chain length (t): %d\n\n", t);

    Password * passwords;

    long size= passwordNumber*t;

    initPasswordArray(&passwords, size, 0);

    printf("Password to be stored in ram: %ld\n", size);

    printf("Number of columns (t): %d\n\n", t);

    printf("mt: %ld\n", mt);

    // Reading all startpoints
    for (long i = 0; i < passwordNumber; i += 1) {
        fread(passwords[i].bytes, pwd_length, 1, (FILE *) fp);
    }

    printf("Copy: ");
    printf("Should be first: ");
    printPassword(&passwords[0]);
    printf(" Should be last: ");
    printPassword(&passwords[passwordNumber-1]);
    printf("\n");

    // Close the start file
    fclose(fp);

    int offset = passwordNumber-1;
    printf("Generated chain for the startpoint number %d: \n", offset);

    //i<t car on ne regarde pas les endpoints
    for(int i=0; i<t;i++) {
        // Copy startpoints before launching kernel on it
        for (long j = 0; j < passwordNumber; j++) {
            memcpy(&passwords[i*passwordNumber+j], &passwords[j], sizeof(Password));
        }
        printf("\n-----\n");
        printf("Before: ");
        printPassword(&(passwords[(i*passwordNumber)+offset]));
        printf("\n");
        printf("%d: ", i);
        generateChains(&(passwords[i*passwordNumber]), passwordNumber, 1, i,
                       false, THREAD_PER_BLOCK, false, false, NULL, pwd_length, start_path, end_path);
        printf("After: ");
        printPassword(&(passwords[(i*passwordNumber)+offset]));
        printf("\n-----\n");
    }

    printf("Generation done\n");

    int nbFound = 0;
    int nbNotFound = 0;

    Password * result = (Password *)malloc(pwd_length);

    long domain = pow(CHARSET_LENGTH, pwd_length);

    for(int i=0; i<domain; i++){
        // Generate one password
        long counter = i;
        for (int b = 0; b<pwd_length; b++) {
            (*result).bytes[b] = charset[counter % CHARSET_LENGTH];
            counter /= CHARSET_LENGTH;
        }

        for(int k=0; k < size; k++){
            if(memcmp(&(*result).bytes, &(passwords[k].bytes), pwd_length) == 0){
                nbFound++;
                /*
                printf("Trouvé! ");
                for(int q=0; q<pwd_length; q++){
                    printf("%c", (*result).bytes[q]);
                }
                printf("\n");
                */
                break;
            }else if (k == size-1){
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
    printf("Number of passwords found: %d\n", nbFound);
    printf("Number of passwords not found: %d\n", nbNotFound);
    printf("Coverage: %f\n", ((double)(double)nbFound / (double)domain) * 100);

    cudaFreeHost(passwords);

    return 0;
}