// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <nvtx3/nvToolsExt.h>
// #include "array.h"
// #include "shellsort.h"

// // Algorithm parameters
// const int seed = 0;
// const int warmup = 2;
// const int iterations = 5;
// const size_t arraySize = 1'000'000;
// typedef void CUDASort(const char*, int*, int*, size_t, bool);

// void runSort(const char* sortName, CUDASort cudaSort, int* h_inputArray, int* h_outputArray) {
//     // Warmup the GPU
//     for (int i = 0; i < warmup; i++) cudaSort(sortName, h_inputArray, h_outputArray, arraySize, false);
    
//     // Create CUDA events for timing
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     float totalTime = 0.0f;

//     for (int i = 0; i < iterations; i++) {
//         // Start recording
//         cudaEventRecord(start, NULL);
//         nvtxRangePush(sortName);

//         // Run parallel sorting algorithm
//         cudaSort(sortName, h_inputArray, h_outputArray, arraySize, true);

//         // Stop recording
//         nvtxRangePop();
//         cudaEventRecord(stop, NULL);
//         cudaEventSynchronize(stop);

//         // Calculate elapsed time
//         float milliseconds = 0;
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         totalTime += milliseconds;

//         // Check if the array is sorted correctly
//         if (!checkArraySorted(h_inputArray, h_outputArray, arraySize)) {
//             fprintf(stderr, "Error (%s): Array not sorted correctly\n", sortName);
//             exit(EXIT_FAILURE);
//         }
//     }

//     // Compute and print the average time per iteration
//     float avgTime = totalTime / iterations;
//     printf("Average Time (%s): %f ms\n", sortName, avgTime);

//     // Destroy CUDA events
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
// }

// int main() {
//     // Allocate and initialize arrays
//     size_t arrayByteSize = arraySize * sizeof(int);
//     int* h_inputArray = (int*)malloc(arrayByteSize);
//     int* h_outputArray = (int*)malloc(arrayByteSize);
//     generateRandomArray(h_inputArray, arraySize, seed);

//     // Run sorting algorithms
//     runSort("shellsort", GPUSort, h_inputArray, h_outputArray);
//     runSort("thrustsort", GPUSort, h_inputArray, h_outputArray);

//     // Free host memory
//     free(h_inputArray);
//     free(h_outputArray);
//     return 0;
// }

// void GPUSort(const char* sortName, int* h_inputArray, int* h_outputArray, size_t arraySize, bool saveOutput) {
//     int* d_array;
//     size_t arrayByteSize = arraySize * sizeof(int);
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_array, arrayByteSize));
//     CHECK_CUDA_ERROR(cudaMemcpyAsync(d_array, h_inputArray, arrayByteSize, cudaMemcpyHostToDevice, stream));

//     if (std::strcmp(sortName, "shellsort") == 0) shellsort(d_array, arraySize, stream);
//     if (std::strcmp(sortName, "thrustsort") == 0) thrustsort(d_array, arraySize, stream);

//     if (saveOutput) CHECK_CUDA_ERROR(cudaMemcpyAsync(h_outputArray, d_array, arrayByteSize, cudaMemcpyDeviceToHost, stream));
//     cudaStreamSynchronize(stream);
//     cudaStreamDestroy(stream);
//     cudaFree(d_array);
// }