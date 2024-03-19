#include <nvtx3/nvToolsExt.h>
#include "hetsort.cuh"

int main(int argc, char* argv[]) {
    int seed = 42;
    size_t arraySize = (argc > 1) ? std::atoi(argv[1]) : 20'000'000;
    size_t bufferSize = (argc > 2) ? std::atoi(argv[2]) : 10;
    printf("Array size: %zu. Array byte size: %zu MB. Buffer size: %zu MB.\n", arraySize, arraySize * sizeof(int) / (1024 * 1024), bufferSize);

    // Get GPU information
    bufferSize = bufferSize * 1024 * 1024;
    std::vector<GPUInfo> gpus = getGPUsInfo(bufferSize * 1.5, true);

    // Allocate and initialize arrays
    int* h_inputArray = (int*)malloc(arraySize * sizeof(int));
    generateRandomArray(h_inputArray, arraySize, seed);
    std::unordered_map<int, int> originalCounts = countElements(h_inputArray, arraySize);

    // Split the array into chunks based on GPU memory availability
    std::vector<std::vector<std::vector<int>>> chunkGroups = splitArray(h_inputArray, arraySize, bufferSize, gpus);

    // Sort each chunk on the GPU
    sortThrust2N(chunkGroups, gpus);

    // Check if each chunk is sorted correctly
    if (checkChunkGroupsSorted(chunkGroups, originalCounts)) printf("Chunks are sorted correctly\n");

    // Perform multi-way merge
    std::vector<int> merged_result = multiWayMerge(chunkGroups);

    // Check if the merged array is sorted correctly
    if (checkArraySorted(merged_result.data(), originalCounts, arraySize)) printf("Array is sorted correctly\n");

    // Clean up
    free(h_inputArray);
    return 0;
}

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