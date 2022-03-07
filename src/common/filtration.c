#include "filtration.h"

int cmpstr(void *v1, void *v2, int len) {
    return memcmp(v1, v2, len);
}

void swap(void *v1, void *v2, long size) {
    char buffer[size];

    memcpy(buffer, v1, size);
    memcpy(v1, v2, size);
    memcpy(v2, buffer, size);
}

void q_sort(void *v, void *m, long size, long left, long right, int (*comp)(void *, void *, int)) {
    void *vt, *v3, *mt, *m3;
    long i, last, mid = (left + right) / 2;
    if (left >= right) {
        return;
    }

    // v left value
    void *vl = (char *) (v + (left * size));
    // v right value
    void *vr = (char *) (v + (mid * size));
    // m left value
    void *ml = (char *) (m + (left * size));
    // m right value
    void *mr = (char *) (m + (mid * size));

    swap(vl, vr, size);
    swap(ml, mr, size);

    last = left;
    for (i = left + 1; i <= right; i++) {

        // vl and vt will have the starting address
        // of the elements which will be passed to
        // comp function.
        vt = (char *) (v + (i * size));
        mt = (char *) (m + (i * size));
        if ((*comp)(vl, vt, size) > 0) {
            ++last;
            v3 = (char *) (v + (last * size));
            m3 = (char *) (m + (last * size));
            swap(vt, v3, size);
            swap(mt, m3, size);
        }
    }
    v3 = (char *) (v + (last * size));
    m3 = (char *) (m + (last * size));
    swap(vl, v3, size);
    swap(ml, m3, size);
    q_sort(v, m, size, left, last - 1, comp);
    q_sort(v, m, size, last + 1, right, comp);
}


long dedup(void *v, void *m, int size, long mt, int (*comp)(void *, void *, int)) {
    long index = 1;
    for (long i = 1; i < mt; i++) {
        void *prev = (char *) (v + ((i - 1) * size));
        void *actual = (char *) (v + (i * size));
        void *mirror = (char *) (m + (i * size));

        if ((*comp)(prev, actual, size) != 0) {
            void *indexed = (char *) (v + (index * size));
            void *indexed_mirror = (char *) (m + (index * size));
            memcpy(indexed, actual, size);
            memcpy(indexed_mirror, mirror, size);

            index++;
        }
    }
    return index;
}

long *filter(const char *start_path, const char *end_path, const char *start_out_path, const char *end_out_path) {
    char buff[255];

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

    // Retrieve the number of points
    long mt;
    fscanf(start_file, "%s", buff);
    sscanf(buff, "%ld", &mt);
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


    long limit = mt * (long) pwd_length;

    // Retrieve the points
    char *startpoints = (char *) malloc(sizeof(char) * limit);
    char *endpoints = (char *) malloc(sizeof(char) * limit);

    char buff_point[pwd_length];

    for (unsigned long i = 0; i < limit; i = i + sizeof(char) * pwd_length) {

        fread(buff_point, pwd_length, 1, (FILE *) start_file);
        for (unsigned long j = i; j < i + pwd_length; j++) {
            startpoints[j] = buff_point[j - i];
        }

        fread(buff_point, pwd_length, 1, (FILE *) end_file);
        for (unsigned long j = i; j < i + pwd_length; j++) {
            endpoints[j] = buff_point[j - i];
        }

    }

    // Close the files
    fclose(start_file);
    fclose(end_file);

    q_sort(endpoints, startpoints, sizeof(char) * pwd_length, 0, mt - 1, (int (*)(void *, void *, int)) (cmpstr));

    long new_len = dedup(endpoints, startpoints, sizeof(char) * pwd_length, mt,
                         (int (*)(void *, void *, int)) (cmpstr));

    // Write new files
    long sp_success = 0;
    long ep_success = 0;

    FILE *sp_out_file = fopen(start_out_path, "wb");
    FILE *ep_out_file = fopen(end_out_path, "wb");

    // Write the header
    fprintf(sp_out_file, "%ld\n", new_len);
    fprintf(ep_out_file, "%ld\n", new_len);
    fprintf(sp_out_file, "%d\n", pwd_length);
    fprintf(ep_out_file, "%d\n", pwd_length);
    fprintf(sp_out_file, "%d\n", t);
    fprintf(ep_out_file, "%d\n", t);


    char *point = (char *) malloc(sizeof(char) * pwd_length);
    for (long i = 0; i < new_len; i++) {
        memcpy(point, startpoints + (i * pwd_length), sizeof(char) * pwd_length);
        fwrite(point, pwd_length, 1, (FILE *) sp_out_file);
        sp_success++;

        memcpy(point, endpoints + (i * pwd_length), sizeof(char) * pwd_length);
        fwrite(point, pwd_length, 1, (FILE *) ep_out_file);
        ep_success++;
    }

    fclose(sp_out_file);
    fclose(ep_out_file);

    long *success = (long *) malloc(sizeof(long) * 4);
    success[0] = mt;
    success[1] = new_len;
    success[2] = sp_success;
    success[3] = ep_success;

    return success;
}
