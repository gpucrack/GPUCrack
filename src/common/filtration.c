#include "filtration.h"

// A heap has current size and array of elements
struct MaxHeap
{
    long size;
    char* array;
};

// A utility function to swap to integers
void swap2(char* a, char* b, int size) {
    char buffer[size];

    memcpy(buffer, a, size);
    memcpy(a, b, size);
    memcpy(b, buffer, size);
}

// The main function to heapify a Max Heap. The function
// assumes that everything under given root (element at
// index idx) is already heapified
void maxHeapify(struct MaxHeap* maxHeap, char* array_spectre, long idx, int pwd_length)
{
    long largest = idx;  // Initialize largest as root
    long left = 2*idx + pwd_length;
    long right = 2*idx + 2*pwd_length;

    // See if left child of root exists and is greater than
    // root
    if (left < maxHeap->size &&
        memcmp(maxHeap->array + left, maxHeap->array + largest, pwd_length) > 0)
        largest = left;

    // See if right child of root exists and is greater than
    // the largest so far
    if (right < maxHeap->size &&
        memcmp(maxHeap->array + right, maxHeap->array + largest, pwd_length) > 0)
        largest = right;

    // Change root, if needed
    if (largest != idx)
    {
        swap2(&maxHeap->array[largest], &maxHeap->array[idx], pwd_length);
        swap2(&array_spectre[largest], &array_spectre[idx], pwd_length);
        maxHeapify(maxHeap, array_spectre, largest, pwd_length);
    }
}

// A utility function to create a max heap of given capacity
struct MaxHeap* createAndBuildHeap(char *array, char * array_spectre, long size, int pwd_length)
{
    long i;
    struct MaxHeap* maxHeap =
            (struct MaxHeap*) malloc(sizeof(struct MaxHeap));
    maxHeap->size = size;   // initialize size of heap
    maxHeap->array = array; // Assign address of first element of array

    // Start from bottommost and rightmost internal mode and heapify all
    // internal modes in bottom up way
    for (i = ((((maxHeap->size)/pwd_length) - 2) / 2); i >= 0; --i)
        maxHeapify(maxHeap, array_spectre, i*pwd_length, pwd_length);
    return maxHeap;
}

// The main function to sort an array of given size
void heapSort(char* array, char* array_spectre, long size, int pwd_length, int debug)
{
    // Build a heap from the input data.
    struct MaxHeap* maxHeap = createAndBuildHeap(array, array_spectre, size, pwd_length);
    // Repeat following steps while heap size is greater than 1.
    // The last element in max heap will be the minimum element
    while (maxHeap->size > pwd_length)
    {
        // The largest item in Heap is stored at the root. Replace
        // it with the last item of the heap followed by reducing the
        // size of heap by 1.
        swap2(&maxHeap->array[0], &maxHeap->array[maxHeap->size - pwd_length], pwd_length);
        swap2(&array_spectre[0], &array_spectre[maxHeap->size - pwd_length], pwd_length);
        maxHeap->size = maxHeap->size - pwd_length;  // Reduce heap size

        // Finally, heapify the root of tree.
        maxHeapify(maxHeap,array_spectre, 0, pwd_length);

        // Print the size of heap if it is a multiple of size/1000
        if (debug && maxHeap->size % (size/100) == 0) printf("%ld\n", maxHeap->size);

    }
}

int cmpstr(void *v1, void *v2, int len) {
    return memcmp(v1, v2, len);
}

void swap(void *v1, void *v2, long size) {
    char buffer[size];

    memcpy(buffer, v1, size);
    memcpy(v1, v2, size);
    memcpy(v2, buffer, size);
}

void q_sort(void *v, void *m, long size, long long left, long long right, int (*comp)(void *, void *, int)) {
    char *vt, *v3, *mt, *m3;
    long long i, last, mid = (long long)(((long long)left + (long long)right) / (long long)2);
    if (left >= right) {
        return;
    }

    // v left value
    char *vl = (char *) (v + (long long)((long long)left * (long long)size));
    // v right value
    char *vr = (char *) (v + (long long)((long long)mid * (long long)size));
    // m left value
    char *ml = (char *) (m + (long long)((long long)left * (long long)size));
    // m right value
    char *mr = (char *) (m + (long long)((long long)mid * (long long)size));

    swap(vl, vr, size);
    swap(ml, mr, size);

    last = left;
    for (i = left + 1; i <= right; i++) {

        // vl and vt will have the starting address
        // of the elements which will be passed to
        // comp function.
        vt = (char *) (v + (long long)((long long)i * (long long)size));
        mt = (char *) (m + (long long)((long long)i * (long long)size));
        if ((*comp)(vl, vt, size) > 0) {
            ++last;
            v3 = (char *) (v + (long long)((long long)last * (long long)size));
            m3 = (char *) (m + (long long)((long long)last * (long long)size));
            swap(vt, v3, size);
            swap(mt, m3, size);
        }
    }
    v3 = (char *) (v + (long long)((long long)last * (long long)size));
    m3 = (char *) (m + (long long)((long long)last * (long long)size));
    swap(vl, v3, size);
    swap(ml, m3, size);

    q_sort(v, m, size, left, last - 1, comp);
    q_sort(v, m, size, last + 1, right, comp);
}


long long int dedup(char *v, char *m, int size, long long int mt, int (*comp)(void *, void *, int)) {
    long long index = 1;
    for (long long i = 1; i < mt; i++) {
        char *prev = (char *) (v + (long long)((long long)(i - 1) * (long long)size));
        char *actual = (char *) (v + (long long)((long long)i * (long long)size));
        char *mirror = (char *) (m + (long long)((long long)i * (long long)size));

        if ((*comp)(prev, actual, size) != 0) {
            char *indexed = (char *) (v + (long long)((long long)index * (long long)size));
            char *indexed_mirror = (char *) (m + (long long)((long long)index * (long long)size));
            memcpy(indexed, actual, size);
            memcpy(indexed_mirror, mirror, size);

            index++;
        }
    }
    return index;
}

long *
filter(char *start_path, char *end_path, const char *start_out_path, const char *end_out_path, int numberOfPasses,
       unsigned long long batchSize, char *path, unsigned long long passwordMemory, bool debug) {

    if (batchSize*2 > passwordMemory) numberOfPasses *= 2;
    batchSize = (batchSize/2) + 1;

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

    unsigned long long totalNewLen = 0;
    unsigned long long currentPos = 0;

    long sp_success = 0;
    long ep_success = 0;

    char tempStartName[100] = "";
    char tempEndName[100] = "";
    strcat(tempStartName, path);
    strcat(tempStartName, "_temp_start.bin");
    strcat(tempEndName, path);
    strcat(tempEndName, "_temp_end.bin");

    FILE *sp_out_file = fopen(tempStartName, "wb");
    FILE *ep_out_file = fopen(tempEndName, "wb");

    if (sp_out_file == NULL) {
        perror("Can't open start file.");
        printf("%s\n", tempStartName);
        exit(1);
    }

    if (ep_out_file == NULL) {
        perror("Can't open end file.");
        printf("%s\n", tempEndName);
        exit(1);
    }

    // Write the header
    fprintf(sp_out_file, "%llu\n", batchSize);
    fprintf(ep_out_file, "%llu\n", batchSize);
    fprintf(sp_out_file, "%d\n", pwd_length);
    fprintf(ep_out_file, "%d\n", pwd_length);
    fprintf(sp_out_file, "%d\n", t);
    fprintf(ep_out_file, "%d\n", t);

    for(int q=0; q<numberOfPasses; q++) {

        // Print batch number
        if (debug) printf("Batch %d\n", q);

        unsigned long long limit = (unsigned long long)batchSize * (unsigned long long)pwd_length;

        // Retrieve the points
        char *startpoints = (char *) malloc(sizeof(char) * limit);
        char *endpoints = (char *) malloc(sizeof(char) * limit);

        char buff_point[pwd_length];

        for (unsigned long long i = 0; i < limit; i = i + sizeof(char) * pwd_length) {

            fread(buff_point, pwd_length, 1, (FILE *) start_file);
            for (unsigned long long j = i; j < i + pwd_length; j++) {
                startpoints[j] = buff_point[j - i];
            }

            fread(buff_point, pwd_length, 1, (FILE *) end_file);
            for (unsigned long long j = i; j < i + pwd_length; j++) {
                endpoints[j] = buff_point[j - i];
            }

        }

        // Display a message if size % 10 == 0

        //printf("\n\nPass %d: %llu points\n", q, batchSize);

        //q_sort(endpoints, startpoints, sizeof(char) * pwd_length, 0, batchSize - 1, (int (*)(void *, void *, int)) (cmpstr));
        heapSort(endpoints, startpoints, limit, pwd_length, debug);

        unsigned long long new_len = dedup(endpoints, startpoints, sizeof(char) * pwd_length, batchSize,
                             (int (*)(void *, void *, int)) (cmpstr));

        totalNewLen += new_len;

        // Write points
        char *point = (char *) malloc(sizeof(char) * pwd_length);
        for (unsigned long long i = 0; i < new_len; i++) {
            memcpy(point, startpoints + (i * pwd_length), sizeof(char) * pwd_length);
            fwrite(point, pwd_length, 1, (FILE *) sp_out_file);
            sp_success++;

            memcpy(point, endpoints + (i * pwd_length), sizeof(char) * pwd_length);
            fwrite(point, pwd_length, 1, (FILE *) ep_out_file);
            ep_success++;
        }

        free(startpoints);
        free(endpoints);

        currentPos += batchSize;
    }

    // Close the files
    fclose(start_file);
    fclose(end_file);

    // Close final file and then reopen for final sort and update the position to the start
    fclose(sp_out_file);
    fclose(ep_out_file);

    sp_out_file = fopen(tempStartName, "rb");
    ep_out_file = fopen(tempEndName, "rb");

    if (sp_out_file == NULL) {
        perror("Can't open start file.");
        exit(1);
    }

    if (ep_out_file == NULL) {
        perror("Can't open end file.");
        exit(1);
    }

    // Skipping headers, we already know the values
    fgets(buff, 255, (FILE *) sp_out_file);
    fgets(buff, 255, (FILE *) sp_out_file);
    fgets(buff, 255, (FILE *) sp_out_file);

    fgets(buff, 255, (FILE *) ep_out_file);
    fgets(buff, 255, (FILE *) ep_out_file);
    fgets(buff, 255, (FILE *) ep_out_file);

    // Final sort
    unsigned long long limit = totalNewLen * (long) pwd_length;

    char *startpoints = (char *) malloc(sizeof(char) * limit);
    char *endpoints = (char *) malloc(sizeof(char) * limit);

    char buff_point[pwd_length];

    for (unsigned long long i = 0; i < limit; i = i + sizeof(char) * pwd_length) {

        fread(buff_point, pwd_length, 1, (FILE *) sp_out_file);
        for (unsigned long long j = i; j < i + pwd_length; j++) {
            startpoints[j] = buff_point[j - i];
        }

        fread(buff_point, pwd_length, 1, (FILE *) ep_out_file);
        for (unsigned long long j = i; j < i + pwd_length; j++) {
            endpoints[j] = buff_point[j - i];
        }

    }

    // We can delete temporary files now
    remove(tempStartName);
    remove(tempEndName);

    unsigned long long new_len;

    // Don't filter if we have only one part
    if (numberOfPasses > 1) {

        //q_sort(endpoints, startpoints, sizeof(char) * pwd_length, 0, totalNewLen - 1, (int (*)(void *, void *, int)) (cmpstr));
        heapSort(endpoints, startpoints, limit, pwd_length, 1);

        new_len = dedup(endpoints, startpoints, sizeof(char) * pwd_length, totalNewLen,
                             (int (*)(void *, void *, int)) (cmpstr));

    }else{
        new_len = totalNewLen;
    }

    sp_out_file = fopen(start_out_path, "wb");
    ep_out_file = fopen(end_out_path, "wb");

    if (sp_out_file == NULL) {
        perror("Can't open start file.");
        exit(1);
    }

    if (ep_out_file == NULL) {
        perror("Can't open end file.");
        exit(1);
    }

    // Write the header
    fprintf(sp_out_file, "%llu\n", new_len);
    fprintf(ep_out_file, "%llu\n", new_len);
    fprintf(sp_out_file, "%d\n", pwd_length);
    fprintf(ep_out_file, "%d\n", pwd_length);
    fprintf(sp_out_file, "%d\n", t);
    fprintf(ep_out_file, "%d\n", t);

    // Write points
    char *point = (char *) malloc(sizeof(char) * pwd_length);

    for (unsigned long long i = 0; i < new_len; i++) {
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
