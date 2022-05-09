#include "filtrationHT.cuh"
using namespace std;

unsigned long long
finalFilter(FILE *tempStart, FILE *tempEnd, unsigned long long tempLen, FILE *startFile, FILE *endFile, int pwd_length,
            int t) {

    map<PasswordHT , PasswordHT, less<>> map;

    unsigned long long limit = (unsigned long long)tempLen * (unsigned long long)pwd_length;

    // Store all points inside hash table
    for (unsigned long long i = 0; i < limit; i += (sizeof(char) * pwd_length)) {

        PasswordHT end;
        PasswordHT start;
        fread(end.bytes, pwd_length, 1, (FILE *) tempEnd);
        fread(start.bytes, pwd_length, 1, (FILE *) tempStart);

        map.insert(pair<PasswordHT, PasswordHT>(end,start));
    }

    unsigned long long newLen = map.size();

    // Write the header
    fprintf(startFile, "%llu\n", newLen);
    fprintf(endFile, "%llu\n", newLen);
    fprintf(startFile, "%d\n", pwd_length);
    fprintf(endFile, "%d\n", pwd_length);
    fprintf(startFile, "%d\n", t);
    fprintf(endFile, "%d\n", t);

    // Then write the sorted points inside the final files
    for (auto it = map.begin(); it != map.end(); ++it) {

        PasswordHT endPoint = it->first;
        PasswordHT startPoint = it->second;

        fwrite(startPoint.bytes, pwd_length, 1, (FILE *) startFile);
        fwrite(endPoint.bytes, pwd_length, 1, (FILE *) endFile);
    }

    // Destroy the map
    map.clear();

    return newLen;
}

unsigned long long
filterBatch(FILE *startFile, FILE *endFile, unsigned long long batchSize, int pwd_length, FILE *tempStart,
            FILE *tempEnd) {

    map<PasswordHT , PasswordHT, less<>> map;

    unsigned long long limit = (unsigned long long)batchSize * (unsigned long long)pwd_length;

    // Store all points inside hash table
    for (unsigned long long i = 0; i < limit; i += (sizeof(char) * pwd_length)) {

        PasswordHT end;
        PasswordHT start;
        fread(end.bytes, pwd_length, 1, (FILE *) endFile);
        fread(start.bytes, pwd_length, 1, (FILE *) startFile);

        map.insert(pair<PasswordHT, PasswordHT>(end,start));
    }

    // Then write the sorted points inside a temporary file
    for (auto it = map.begin(); it != map.end(); ++it) {

        PasswordHT endPoint = it->first;
        PasswordHT startPoint = it->second;

        fwrite(startPoint.bytes, pwd_length, 1, (FILE *) tempStart);
        fwrite(endPoint.bytes, pwd_length, 1, (FILE *) tempEnd);
    }

    unsigned long long newLen = map.size();

    // Destroy the map
    map.clear();

    return newLen;
}

long *
filterHT(char *start_path, char *end_path, const char *start_out_path, const char *end_out_path, char *path,
         unsigned long long batchSize, int numberOfPasses) {

    FILE *start_file;
    FILE *end_file;

    start_file = fopen(start_path, "rb");
    end_file = fopen(end_path, "rb");

    if (start_file == NULL) {
        perror("Can't open start file.");
        exit(1);
    }

    if (end_file == NULL) {
        perror("Can't open end file.");
        exit(1);
    }

    char buff[255];

    // Retrieve the number of points
    unsigned long long mt;
    fscanf(start_file, "%s", buff);
    sscanf(buff, "%llu", &mt);
    fgets(buff, 255, (FILE *) start_file);

    // Retrieve the password length
    int pwd_length;
    fgets(buff, 255, (FILE *) start_file);
    sscanf(buff, "%d", &pwd_length);

    // Retrieve the chain length (t)
    int t;
    fgets(buff, 255, (FILE *) start_file);
    sscanf(buff, "%d", &t);

    // just to skip the first 3 rows of end_file (same as start_file)
    fgets(buff, 255, (FILE *) end_file);
    fgets(buff, 255, (FILE *) end_file);
    fgets(buff, 255, (FILE *) end_file);

    // Open temporary file
    char tempStartName[100] = "";
    char tempEndName[100] = "";
    strcat(tempStartName, path);
    strcat(tempStartName, "_temp_start.bin");
    strcat(tempEndName, path);
    strcat(tempEndName, "_temp_end.bin");

    FILE * tempStart = fopen(tempStartName, "wb");
    FILE * tempEnd = fopen(tempEndName, "wb");

    if (tempStart == NULL) {
        perror("Can't open start file.");
        printf("%s\n", tempStartName);
        exit(1);
    }

    if (tempEnd == NULL) {
        perror("Can't open end file.");
        printf("%s\n", tempEndName);
        exit(1);
    }

    unsigned long long tempLen;

    // Filter each batch
    for(int i = 0; i<numberOfPasses; i++) {
        tempLen += filterBatch(start_file, end_file, batchSize,
                    pwd_length, tempStart, tempEnd);
    }

    fclose(tempStart);
    fclose(tempEnd);
    fclose(start_file);
    fclose(end_file);

    FILE * sp_out_file = fopen(start_out_path, "wb");
    FILE * ep_out_file = fopen(end_out_path, "wb");

    if (sp_out_file == NULL) {
        perror("Can't open start file.");
        exit(1);
    }

    if (ep_out_file == NULL) {
        perror("Can't open end file.");
        exit(1);
    }

    tempStart = fopen(tempStartName, "rb");
    tempEnd = fopen(tempEndName, "rb");

    if (tempStart == NULL) {
        perror("Can't open start file.");
        printf("%s\n", tempStartName);
        exit(1);
    }

    if (tempEnd == NULL) {
        perror("Can't open end file.");
        printf("%s\n", tempEndName);
        exit(1);
    }

    // Now do the final filter
    unsigned long long newLen = finalFilter(tempStart, tempEnd, tempLen, sp_out_file, ep_out_file, pwd_length, t);

    fclose(sp_out_file);
    fclose(ep_out_file);
    fclose(tempStart);
    fclose(tempEnd);

    // We can delete temporary files now
    remove(tempStartName);
    remove(tempEndName);

    long *success = (long *) malloc(sizeof(long) * 2);
    success[0] = mt;
    success[1] = newLen;

    return success;

}