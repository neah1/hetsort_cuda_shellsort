#include "hetsort.cuh"

// Algorithm parameters
bool checkSorted = true;
bool bitonicChunks = false;
std::string method, distribution;
size_t arraySize, deviceMemory, iterations, warmup, seed;
typedef void CUDASorter(std::vector<std::vector<std::vector<int>>>&, std::vector<GPUInfo>&);

std::vector<int> runSort(CUDASorter cudaSorter, int* h_inputArray, size_t chunkSize, std::vector<GPUInfo>& gpus) {
    nvtxRangePush("ArraySplit phase");
    std::vector<std::vector<std::vector<int>>> chunkGroups = splitArray(h_inputArray, arraySize, chunkSize, gpus, bitonicChunks);
    nvtxRangePop();

    nvtxRangePush("Kernel phase");
    cudaSorter(chunkGroups, gpus);
    nvtxRangePop();

    nvtxRangePush("Merge phase");
    std::vector<int> h_outputArray = multiWayMerge(chunkGroups, arraySize);
    nvtxRangePop();

    return h_outputArray;
}

void runSortingAlgorithm(CUDASorter cudaSorter, int* h_inputArray, size_t chunkSize, std::vector<GPUInfo>& gpus) {
    // Warmup the GPU
    for (int i = 0; i < warmup; i++) 
        if (method.find("Kernel") != std::string::npos)
            sortKernel(method, h_inputArray, arraySize, gpus);
        else
            runSort(cudaSorter, h_inputArray, chunkSize, gpus);

    for (int i = 0; i < iterations; i++) {
        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        
        nvtxRangePush("Sorting algorithm");
        std::vector<int> h_outputArray;
        if (method.find("Kernel") != std::string::npos)
            h_outputArray = sortKernel(method, h_inputArray, arraySize, gpus);
        else
            h_outputArray = runSort(cudaSorter, h_inputArray, chunkSize, gpus);
        nvtxRangePop();

        // Stop timing
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        printf("Iteration %d: %lld ms\n", i, duration);

        // Check if the array is sorted correctly
        if (checkSorted) {
            std::unordered_map<int, int> originalCounts = countElements(h_inputArray, arraySize);
            checkArraySorted(h_outputArray.data(), originalCounts, arraySize);
        }
    }
}

CUDASorter* selectSortingMethod(size_t& bufferCount, size_t& chunkSize) {
    CUDASorter* cudaSorter = nullptr;
    deviceMemory = deviceMemory * 1024 * 1024;
    if (method == "thrustsort2N") {
        cudaSorter = sortThrust2N;
        bufferCount = 2;
    } else if (method == "thrustsort3N") {
        cudaSorter = sortThrust3N;
        bufferCount = 3;
    } else if (method == "thrustsortInplace") {
        cudaSorter = sortThrustInplace;
        bufferCount = 2;
    } else if (method == "sortThrustInplaceMemcpy") {
        cudaSorter = sortThrustInplaceMemcpy;
        bufferCount = 2;
    } else if (method == "shellsort") {
        cudaSorter = sortShell;
        bufferCount = 1;
        bitonicChunks = true;
    } else if (method == "shellsort2N") {
        cudaSorter = sortShell2N;
        bufferCount = 2;
        bitonicChunks = true;
    } else if (method == "shellsortKernel") {
        bufferCount = 1;
        deviceMemory = nextPowerOfTwo(arraySize) * sizeof(int);
    } else if (method == "thrustsortKernel") {
        bufferCount = 2;
        deviceMemory = 2 * arraySize * sizeof(int);
    } else {
        std::cerr << "Invalid sorting method.\n";
        exit(EXIT_FAILURE);
    }
    chunkSize = (deviceMemory / bufferCount) / sizeof(int);
    return cudaSorter;
}

void benchmark() {
    // Select sorting method
    size_t bufferCount, chunkSize;
    CUDASorter* cudaSorter = selectSortingMethod(bufferCount, chunkSize);

    nvtxRangePush("GPU information");
    std::vector<GPUInfo> gpus = getGPUsInfo(deviceMemory, bufferCount);
    nvtxRangePop();

    nvtxRangePush("Generate array");
    int* h_inputArray;
    cudaMallocHost((void**)&h_inputArray, arraySize * sizeof(int));
    generateRandomArray(h_inputArray, arraySize, seed, distribution);
    nvtxRangePop();

    // Run sorting algorithm
    runSortingAlgorithm(cudaSorter, h_inputArray, chunkSize, gpus);

    // Clean up
    cudaFreeHost(h_inputArray);
}

int main(int argc, char* argv[]) {
    method = (argc > 1) ? argv[1] : "shellsort";
    distribution = (argc > 2) ? argv[2] : "uniform";
    arraySize = (argc > 3) ? std::atoi(argv[3]) : 10'000'000;
    deviceMemory = (argc > 4) ? std::atoi(argv[4]) : 100;
    
    iterations = (argc > 5) ? std::atoi(argv[5]) : 1;
    warmup = (argc > 6) ? std::atoi(argv[6]) : 0;
    checkSorted = (argc > 7) ? std::atoi(argv[7]) : 1;
    seed = (argc > 8) ? std::atoi(argv[8]) : 42;

    std::string label = "Method: " + method + ", Distribution: " + distribution + 
                        ", Array Size: " + std::to_string(arraySize) + 
                        ", Array Byte Size: " + std::to_string(arraySize * sizeof(int) / (1024 * 1024)) + " MB" +
                        ", Device Memory: " + std::to_string(deviceMemory) + " MB" +
                        ", Iterations: " + std::to_string(iterations) + ", Warmup: " + std::to_string(warmup) + "\n";
    printf(label.c_str());

    benchmark();
    return 0;
}
