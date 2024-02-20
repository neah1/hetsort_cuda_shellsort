#ifndef SHELLSORT_H
#define SHELLSORT_H
#include <cuda_runtime.h>

__global__ void shellsortKernel(int* d_array, size_t arraySize, int increment);
void shellsort(int* d_array, size_t arraySize);
void thrustsort(int* d_array, size_t arraySize);
void GPUSort(const char* sortName, int* h_inputArray, int* h_outputArray, size_t arraySize, bool saveOutput);

#endif
