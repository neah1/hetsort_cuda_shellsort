#include <cuda.h>
#include <cuda_runtime.h>

__global__ void shellsortIncrement(int *array, int arraySize, int increment) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int temp;
    int j;

    if (index >= increment) return;

    for (int i = index; i < arraySize; i += increment) {
        temp = array[i];
        for (j = i; j >= increment && array[j - increment] > temp; j -= increment) {
            array[j] = array[j - increment];
        }
        array[j] = temp;
    }
}

void parallelShellsort(int *array, int arraySize) {
    // const int increments[] = {5, 3, 1}; // Shell's original sequence
    const int increments[] = {1750, 701, 301, 132, 57, 23, 10, 4, 1}; // Increment sequence from Ciura (2001)
    const int numIncrements = sizeof(increments) / sizeof(increments[0]);

    const int numThreads = 256;
    const int numBlocks = (arraySize + numThreads - 1) / numThreads;

    for (int j = 0; j < numIncrements; j++) {
        shellsortIncrement<<<numBlocks, numThreads>>>(array, arraySize, increments[j]);
        cudaDeviceSynchronize();
    }
    
}