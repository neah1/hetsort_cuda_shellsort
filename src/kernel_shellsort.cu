#include "kernel.cuh"

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

void shellsort(int* d_array, size_t arraySize, cudaStream_t stream) {
    // Increment sequence from Ciura (2001), Shell's original sequence: {5, 3, 1}
    int increments[] = {1750, 701, 301, 132, 57, 23, 10, 4, 1};
    int numThreads = 256;
    int numBlocks = (arraySize + numThreads - 1) / numThreads;

    for (int j = 0; j < sizeof(increments) / sizeof(increments[0]); j++) {
        shellsortKernel<<<numBlocks, numThreads, 0, stream>>>(d_array, arraySize, increments[j]);
    }
}

