#ifndef SHELLSORT_H
#define SHELLSORT_H
#include <cuda_runtime.h>

__global__ void shellsortIncrement(int* array, int arraySize, int increment);
void parallelShellsort(int* array, int arraySize);
void thrustSort(int* array, int arraySize);

#endif
