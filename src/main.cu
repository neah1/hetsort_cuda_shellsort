#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "benchmark.h"

__global__ void parallelShellSort(int *array, int arraySize, int increment);

int main()
{
    // Generate input array
    const int seed = 0;
    int iterations = 10;
    const int arraySize = 10'000;
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

    // Warm up the GPU
    for (int i = 0; i < 10; i++)
    {
        parallelShellSort<<<numBlocks, numThreads>>>(d_inputArray, arraySize, 1);
        cudaDeviceSynchronize();
    }

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0f;
    for (int i = 0; i < iterations; i++)
    {
        cudaEventRecord(start, NULL);

        // Run parallel shell-sort
        for (int i = 0; i < sizeof(increments) / sizeof(increments[0]); i++)
        {
            int increment = increments[i];
            parallelShellSort<<<numBlocks, numThreads>>>(d_inputArray, arraySize, increment);
            cudaDeviceSynchronize(); // Ensure kernel execution is finished before next iteration
        }

        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);

        // Calculate the duration and add it to total time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;

        // Verify that the array is sorted
        CHECK_CUDA_ERROR(cudaMemcpy(h_outputArray, d_inputArray, arrayByteSize, cudaMemcpyDeviceToHost));
        if (!checkArraySorted(h_inputArray, h_outputArray, arraySize))
        {
            fprintf(stderr, "Error: Array not sorted correctly\n");
            exit(EXIT_FAILURE);
        }
    }

    // Compute the average time per iteration
    float avgTime = totalTime / iterations;
    printf("Average Time for Sorting: %f ms\n", avgTime);

    // Free host and device memory
    free(h_inputArray);
    free(h_outputArray);
    cudaFree(d_inputArray);

    return 0;
}
