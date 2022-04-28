#ifndef GPU_CRACK_MPIGENERATETABLETEST_CUH
#define GPU_CRACK_MPIGENERATETABLETEST_CUH

#include "../commons.cuh"

#include <mpi.h>
#include <iostream>

#define BATCH_SIZE 3

#define MASTER 0
#define WORK_TAG 1
#define TERMINATE_TAG 2

#define DEBUG_MASTER 1
#define DEBUG_WORKER 1

void nextTask(long task[2], long m0, long batch_size);
void master();
void slave();
void debug(char *str, ...);

#endif //GPU_CRACK_MPIGENERATETABLETEST_CUH
