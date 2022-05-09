#ifndef GPU_CRACK_FILTRATIONHT_CUH
#define GPU_CRACK_FILTRATIONHT_CUH

#include <map>
#include <string>
#include "../gpu/constants.cuh"

struct PasswordHT {
    char bytes[PASSWORD_LENGTH];

    bool operator==(const PasswordHT &p) const {
        return (strncmp(bytes, p.bytes, PASSWORD_LENGTH) == 0);
    }

    bool operator<(const PasswordHT &p) const {
        return (strncmp(bytes, p.bytes, PASSWORD_LENGTH) < 0);
    }
};

long *
filterHT(char *start_path, char *end_path, const char *start_out_path, const char *end_out_path, char *path,
         unsigned long long batchSize, int numberOfPasses);

unsigned long long
filterBatch(FILE *startFile, FILE *endFile, unsigned long long batchSize, int pwd_length, FILE *tempStart,
            FILE *tempEnd);

unsigned long long
finalFilter(FILE *tempStart, FILE *tempEnd, unsigned long long tempLen, FILE *startFile, FILE *endFile, int pwd_length,
            int t);

#endif //GPU_CRACK_FILTRATIONHT_CUH
