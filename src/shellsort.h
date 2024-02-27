#ifndef SHELLSORT_H
#define SHELLSORT_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

__global__ void shellsortKernel(int* d_array, size_t arraySize, int increment);
void shellsort(int* d_array, size_t arraySize, cudaStream_t stream);
void thrustsort(int* d_array, size_t arraySize, cudaStream_t stream);
void GPUSort(const char* sortName, int* h_inputArray, int* h_outputArray, size_t arraySize, bool saveOutput);

#endif
