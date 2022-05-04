#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Compares two strings.
 * @param v1 the first string to compare.
 * @param v2 the second string.
 * @param len the number of bytes to compare.
 * @return
 */
int cmpstr(void *v1, void *v2, int len);

/**
 * Swaps two elements.
 * @param v1 the first element to swap.
 * @param v2 the second element.
 * @param size the number of bytes to swap.
 */
void swap(void *v1, void *v2, long size);

/**
 * Sorts an array.
 * @param v array of elements to sort.
 * @param m (mirror) array of elements to sort, like v.
 * @param size the number of elements in sub-array (pwd_len).
 * @param left start of array.
 * @param right end of array.
 * @param comp pointer to the comparison function.
 */
void q_sort(void *v, void *m, long size, long long left, long long right, int (*comp)(void *, void *, int));

/**
 * Deletes all duplicates in an array.
 * @param v array of elements to deduplicate.
 * @param m (mirror) array of elements to deduplicate, like v.
 * @param size the number of elements in sub-array (pwd_len).
 * @param mtthe size of v.
 * @param comp a pointer to the comparison function.
 * @return the number of startpoints written after deduplication
 */
long long int dedup(char *v, char *m, int size, long long int mt, int (*comp)(void *, void *, int));

/**
 * Cleans a table by deleting all of its duplicate endpoints.
 * @param start_path the path to the startpoint file.
 * @param end_path the path to the startpoint file.
 * @param start_out_path the path to the clean startpoint file that will be generated.
 * @param end_out_path the path to the clean startpoint file that will be generated.
 * @return a long array containing :
 *          - the initial number of endpoints;
 *          - the new number of unique points;
 *          - the number of startpoints written after deduplication;
 *          - the number of endpoints written after deduplication.
 */
long *
filter(char *start_path, char *end_path, const char *start_out_path, const char *end_out_path, int numberOfPasses,
       unsigned long long batchSize, char *path, unsigned long long passwordMemory);