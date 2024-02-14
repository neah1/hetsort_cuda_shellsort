#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "array.h"

// Algorithm parameters
const int seed = 0;
const int warmup = 2;
const int iterations = 5;
const int arraySize = 1'000;
const size_t arrayByteSize = arraySize * sizeof(int);

void warmUpGPU(int* d_array, int* h_inputArray) {
    for (int i = 0; i < warmup; i++) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_array, h_inputArray, arrayByteSize, cudaMemcpyHostToDevice));
        parallelShellsort(d_array, arraySize);
        cudaDeviceSynchronize();
    }
}

void runSort(int* d_array, int* h_inputArray, int* h_outputArray) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0f;
    for (int i = 0; i < iterations; i++) {
        // Copy unsorted array to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_array, h_inputArray, arrayByteSize, cudaMemcpyHostToDevice));

        // Start recording
        cudaEventRecord(start, NULL);
        nvtxRangePush("Shellsort");

        // Run parallel shell-sort for each increment
        parallelShellsort(d_array, arraySize);

        // Stop recording
        nvtxRangePop();
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;

        // Copy sorted array back to host and verify
        CHECK_CUDA_ERROR(cudaMemcpy(h_outputArray, d_array, arrayByteSize, cudaMemcpyDeviceToHost));
        if (!checkArraySorted(h_inputArray, h_outputArray, arraySize)) {
            fprintf(stderr, "Error: Array not sorted correctly\n");
            exit(EXIT_FAILURE);
        }
    }

    // Compute and print the average time per iteration
    float avgTime = totalTime / iterations;
    printf("Average Time for Sorting: %f ms\n", avgTime);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Allocate and initialize arrays
    int* d_array;
    int* h_inputArray = (int*)malloc(arrayByteSize);
    int* h_outputArray = (int*)malloc(arrayByteSize);
    generateRandomArray(h_inputArray, arraySize, seed);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_array, arrayByteSize));

    // Run shellsort
    warmUpGPU(d_array, h_inputArray);
    runSort(d_array, h_inputArray, h_outputArray);

    // Free host and device memory
    cudaFree(d_array);
    free(h_inputArray);
    free(h_outputArray);
    return 0;
}