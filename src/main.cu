#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "array.h"
#include "shellsort.h"

// Algorithm parameters
const int seed = 0;
const int warmup = 2;
const int iterations = 5;
const int arraySize = 1'000;

typedef void CUDASort(int*, int);
const size_t arrayByteSize = arraySize * sizeof(int);

void runSort(const char* sortName, CUDASort cudaSort, int* d_array, int* h_inputArray, int* h_outputArray) {
    // Warmup the GPU
    for (int i = 0; i < warmup; i++) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_array, h_inputArray, arrayByteSize, cudaMemcpyHostToDevice));
        cudaSort(d_array, arraySize);
    }
    
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
        nvtxRangePush(sortName);

        // Run parallel sorting algorithm
        cudaSort(d_array, arraySize);

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
            fprintf(stderr, "Error (%s): Array not sorted correctly\n", sortName);
            exit(EXIT_FAILURE);
        }
    }

    // Compute and print the average time per iteration
    float avgTime = totalTime / iterations;
    printf("Average Time (%s): %f ms\n", sortName, avgTime);

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

    // Run sorting algorithms
    runSort("Shellsort", parallelShellsort, d_array, h_inputArray, h_outputArray);
    runSort("Thrustsort", thrustSort, d_array, h_inputArray, h_outputArray);

    // Free host and device memory
    cudaFree(d_array);
    free(h_inputArray);
    free(h_outputArray);
    return 0;
}