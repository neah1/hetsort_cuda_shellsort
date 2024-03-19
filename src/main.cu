#include "hetsort.cuh"

// Algorithm parameters
std::string method = "thrustsortInplace";
size_t arraySize = 20'000'000;
size_t deviceMemory = 10;
size_t blockSize = 1;
const int seed = 0;

typedef void CUDASorter(std::vector<std::vector<std::vector<int>>>&, std::vector<GPUInfo>&, size_t);

int main(int argc, char* argv[]) {
    method = (argc > 1) ? argv[1] : method;
    arraySize = (argc > 2) ? std::atoi(argv[2]) : arraySize;
    deviceMemory = (argc > 3) ? std::atoi(argv[3]) : deviceMemory;
    blockSize = (argc > 4) ? std::atoi(argv[4]) : blockSize;
    std::cout << "Method: " << method << ". Array size: " << arraySize << ". Array byte size: " << arraySize * sizeof(int) / (1024 * 1024) 
        << " MB. Device memory: " << deviceMemory << " MB. Block size: " << blockSize << " MB.\n";

    // Convert to MB
    deviceMemory = deviceMemory * 1024 * 1024;
    blockSize = blockSize * 1024 * 1024;

    // Select sorting method
    size_t bufferCount = 2;
    CUDASorter* cudaSorter;
    if (method == "thrustsort2N") {
        cudaSorter = sortThrust2N;
    // } else if (method == "thrustsort3N") {
    } else if (method == "thrustsortInplace") {
        cudaSorter = sortThrustInplace;
    } else if (method == "shellsort") {
        bufferCount = 1;
        cudaSorter = sortShell;
    } else if (method == "shellsort2N") {
        cudaSorter = sortShell2N;
    } else {
        printf("Invalid sorting method.\n");
        return 1;
    }

    // Calculate chunk size
    size_t chunkSize = (deviceMemory / bufferCount) / sizeof(int);
    if (method.find("thrust") == 0) chunkSize = chunkSize * 0.80;

    // Get GPU information
    std::vector<GPUInfo> gpus = getGPUsInfo(deviceMemory, bufferCount);

    // Allocate and initialize arrays
    int* h_inputArray = (int*)malloc(arraySize * sizeof(int));
    generateRandomArray(h_inputArray, arraySize, seed);
    std::unordered_map<int, int> originalCounts = countElements(h_inputArray, arraySize);

    // Split the array into chunks based on GPU memory availability
    std::vector<std::vector<std::vector<int>>> chunkGroups = splitArray(h_inputArray, arraySize, chunkSize, gpus);

    // Sort each chunk on the GPU
    cudaSorter(chunkGroups, gpus, blockSize);

    // Check if each chunk is sorted correctly
    checkChunkGroupsSorted(chunkGroups, originalCounts);

    // Perform multi-way merge
    std::vector<int> merged_result = multiWayMerge(chunkGroups);

    // Check if the merged array is sorted correctly
    checkArraySorted(merged_result.data(), originalCounts, arraySize);

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