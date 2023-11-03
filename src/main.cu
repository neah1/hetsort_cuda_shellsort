#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "benchmark.h"

__global__ void parallelShellSort(int *array, int arraySize, int increment);

int main()
{
    // Generate input array
    const int seed = 0;
    const int arraySize = 5000;
    size_t arrayByteSize = arraySize * sizeof(int);
    int *h_inputArray = (int *)malloc(arrayByteSize);
    int *h_outputArray = (int *)malloc(arrayByteSize);
    generateRandomArray(h_inputArray, arraySize, seed);

    // Allocate device memory
    int *d_inputArray;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_inputArray, arrayByteSize));
    CHECK_CUDA_ERROR(cudaMemcpy(d_inputArray, h_inputArray, arrayByteSize, cudaMemcpyHostToDevice));

    // Set kernel parameters
    // int increments[] = {1750, 701, 301, 132, 57, 23, 10, 4, 1}; // Increment sequence from Ciura (2001)
    int increments[] = {5, 3, 1}; // Shell's original sequence
    int numThreads = 1;           // 256
    int numBlocks = (arraySize + numThreads - 1) / numThreads;

    // Run parallel shell-sort
    for (int i = 0; i < sizeof(increments) / sizeof(increments[0]); i++)
    {
        int increment = increments[i];
        parallelShellSort<<<numBlocks, numThreads>>>(d_inputArray, arraySize, increment);
        cudaDeviceSynchronize(); // Ensure kernel execution is finished before next iteration
    }

    // Copy sorted array back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_outputArray, d_inputArray, arrayByteSize, cudaMemcpyDeviceToHost));

    // Print sorted array
    // printArray(h_inputArray, arraySize);
    // printArray(h_outputArray, arraySize);
    if (checkArraySorted(h_inputArray, h_outputArray, arraySize))
    {
        printf("\nSORTED\n");
    }
    else
    {
        printf("\nWRONG\n");
    }

    // Free host and device memory
    free(h_inputArray);
    free(h_outputArray);
    cudaFree(d_inputArray);

    return 0;
}
