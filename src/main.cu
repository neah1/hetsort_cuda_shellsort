#include "hetsort.cuh"

// Algorithm parameters
std::string method = "shellsort";
std::string distribution = "uniform";
size_t arraySize = 10'000'000;
size_t deviceMemory = 2;
const int seed = 42;

typedef void CUDASorter(std::vector<std::vector<std::vector<int>>>&, std::vector<GPUInfo>&);

int main(int argc, char* argv[]) {
    method = (argc > 1) ? argv[1] : method;
    distribution = (argc > 2) ? argv[2] : distribution;
    arraySize = (argc > 3) ? std::atoi(argv[3]) : arraySize;
    deviceMemory = (argc > 4) ? std::atoi(argv[4]) : deviceMemory;
    printf("Method: %s. Distribution: %s. Array size: %zu. Array byte size: %zu MB. Device memory: %zu MB.\n", 
    method.c_str(), distribution.c_str(), arraySize, arraySize * sizeof(int) / (1024 * 1024), deviceMemory);

    // Convert to MB
    deviceMemory = deviceMemory * 1024 * 1024;

    // Select sorting method
    size_t bufferCount = 2;
    CUDASorter* cudaSorter;
    if (method == "thrustsort2N") {
        cudaSorter = sortThrust2N;
    } else if (method == "thrustsort3N") {
        bufferCount = 3;
        cudaSorter = sortThrust3N;
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

    // Allocate pinned memory and initialize arrays
    int* h_inputArray;
    cudaMallocHost((void**)&h_inputArray, arraySize * sizeof(int));
    generateRandomArray(h_inputArray, arraySize, seed, distribution);
    std::unordered_map<int, int> originalCounts = countElements(h_inputArray, arraySize);

    // Split the array into chunks based on GPU memory availability
    std::vector<std::vector<std::vector<int>>> chunkGroups = splitArray(h_inputArray, arraySize, chunkSize, gpus);

    // Sort each chunk on the GPU
    cudaSorter(chunkGroups, gpus);

    // Check if each chunk is sorted correctly
    checkChunkGroupsSorted(chunkGroups, originalCounts);

    // Perform multi-way merge
    std::vector<int> merged_result = multiWayMerge(chunkGroups);

    // Check if the merged array is sorted correctly
    checkArraySorted(merged_result.data(), originalCounts, arraySize);

    // Clean up
    cudaFreeHost(h_inputArray);
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
