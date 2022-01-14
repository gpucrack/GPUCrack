#include "../src/common/reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define REDUCTION_FUNCTION reduceV5 // reduction function to be tested
#define NB_REDUCTIONS 1000000000    // number of reductions to be performed
// Note : The size of the plain texts can be changed in src/common/reduction.h
// (PLAIN_LENGTH)

// Returns the execution time of NB_REDUCTIONS reductions using
// REDUCTION_FUNCTION.
double test_function_speed(char *hash) {
  clock_t start, end;
  double cpu_time_used;
  char *plain = malloc(sizeof(char) * (PLAIN_LENGTH + 1));

  start = clock(); // Start the timer

  for (unsigned long int i = 0; i < NB_REDUCTIONS; i++) {
    REDUCTION_FUNCTION(i, hash, plain);
    // printf("%s\n", plain);
  }

  end = clock(); // End the timer
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

  return cpu_time_used;
}

// Returns the number of collisions (=same reduction returned multiple times)
// for NB_REDUCTIONS reductions using REDUCTION_FUNCTION with the same hash.
long test_function_collision(char *hash) {

  // Allocate memory to store the generated plain texts
  char **plains = malloc(sizeof(char *) * NB_REDUCTIONS);
  for (int i = 0; i < NB_REDUCTIONS; i++) {
    plains[i] = malloc(sizeof(char) * (PLAIN_LENGTH + 1));
  }

  long count = 0;

  for (unsigned long int i = 0; i < NB_REDUCTIONS; i++) {
    REDUCTION_FUNCTION(i, hash, plains[i]);
    // printf("%s\n", plains[i]);
  }

  // Find all duplicate elements in plains
  for (unsigned long int k = 0; k < NB_REDUCTIONS; k++) {
    // printf("%lu/%d\n", k, NB_REDUCTIONS);
    for (unsigned long j = 0; j < NB_REDUCTIONS; j++) {
      // If duplicate found then increment count by 1
      if (k != j && plains[k] == plains[j]) {
        printf("Collision spotted : %s ::: %s\n", plains[k], plains[j]);
        count++;
        break;
      }
    }
  }

  return count;
}

int main() {
    char *hash = "8846f7eeee8fb117ad06bdd830b7586c"; // the hash will remain the
    // same for every iteration
    long collisions = test_function_collision(hash);
    double time = test_function_speed(hash);
    printf("%d reductions of %d characters have been performed in %f "
           "seconds.\nThe character set used contains %d characters.\n",
           NB_REDUCTIONS, PLAIN_LENGTH, time, CHARSET_LENGTH);

    printf("%d reductions of %d characters have been performed in %f seconds "
           "with %ld collision(s).\n",
           NB_REDUCTIONS, PLAIN_LENGTH, time, collisions);

  return 0;
}