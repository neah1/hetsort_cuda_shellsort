#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "array.h"

__global__ void shellsortKernel(int* d_array, size_t arraySize, int increment) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int temp, j;
    
    if (index >= increment) return;
    for (int i = index; i < arraySize; i += increment) {
        temp = d_array[i];
        for (j = i; j >= increment && d_array[j - increment] > temp; j -= increment) {
            d_array[j] = d_array[j - increment];
        }
        d_array[j] = temp;
    }
}

void shellsort(int* d_array, size_t arraySize) {
    // Shell's original sequence: {5, 3, 1}
    // Increment sequence from Ciura (2001)
    int increments[] = {1750, 701, 301, 132, 57, 23, 10, 4, 1}; 
    int numThreads = 256;
    int numBlocks = (arraySize + numThreads - 1) / numThreads;

    for (int j = 0; j < sizeof(increments) / sizeof(increments[0]); j++) {
        shellsortKernel<<<numBlocks, numThreads>>>(d_array, arraySize, increments[j]);
        cudaDeviceSynchronize();
    }
}

void thrustsort(int* d_array, size_t arraySize) {
    thrust::device_ptr<int> array_ptr(d_array);
    thrust::sort(array_ptr, array_ptr + arraySize);
    cudaDeviceSynchronize();
}

void GPUSort(const char* sortName, int* h_inputArray, int* h_outputArray, size_t arraySize, bool saveOutput) {
    int* d_array;
    size_t arrayByteSize = arraySize * sizeof(int);
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_array, arrayByteSize));
    CHECK_CUDA_ERROR(cudaMemcpy(d_array, h_inputArray, arrayByteSize, cudaMemcpyHostToDevice));

    if (std::strcmp(sortName, "shellsort") == 0) shellsort(d_array, arraySize);
    if (std::strcmp(sortName, "thrustsort") == 0) thrustsort(d_array, arraySize);
    
    if (saveOutput) CHECK_CUDA_ERROR(cudaMemcpy(h_outputArray, d_array, arrayByteSize, cudaMemcpyDeviceToHost));
    cudaFree(d_array);
}