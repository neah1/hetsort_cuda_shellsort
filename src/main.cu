#include "hetsort.cuh"

// Algorithm parameters
std::string method, distribution;
size_t arraySize, deviceMemory, seed, warmup, iterations;
typedef void CUDASorter(std::vector<std::vector<std::vector<int>>>&, std::vector<GPUInfo>&);

std::vector<int> runSort(CUDASorter cudaSorter, int* h_inputArray, size_t chunkSize, std::vector<GPUInfo>& gpus) {
    nvtxRangePush("ArraySplit phase");
    std::vector<std::vector<std::vector<int>>> chunkGroups = splitArray(h_inputArray, arraySize, chunkSize, gpus);
    nvtxRangePop();

    nvtxRangePush("Kernel phase");
    cudaSorter(chunkGroups, gpus);
    nvtxRangePop();

    nvtxRangePush("Merge phase");
    std::vector<int> h_outputArray = multiWayMerge(chunkGroups);
    nvtxRangePop();

    return h_outputArray;
}

void runSortingAlgorithm(CUDASorter cudaSorter, int* h_inputArray, size_t chunkSize, std::vector<GPUInfo>& gpus) {
    // Count the number of elements in the input array
    std::unordered_map<int, int> originalCounts = countElements(h_inputArray, arraySize);

    // Warmup the GPU
    for (int i = 0; i < warmup; i++) 
        if (method.find("Kernel") == 0)
            sortKernel(method, h_inputArray, arraySize, gpus);
        else
            runSort(cudaSorter, h_inputArray, chunkSize, gpus);

    for (int i = 0; i < iterations; i++) {
        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        
        nvtxRangePush("Sorting algorithm");
        std::vector<int> h_outputArray;
        if (method.find("Kernel") == 0)
            h_outputArray = sortKernel(method, h_inputArray, arraySize, gpus);
        else
            h_outputArray = runSort(cudaSorter, h_inputArray, chunkSize, gpus);
        nvtxRangePop();

        // Stop timing
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        printf("Iteration %d: %lld ms\n", i, duration);

        // Check if the array is sorted correctly
        if (!checkArraySorted(h_outputArray.data(), originalCounts, arraySize)) {
            fprintf(stderr, "Error (%s): Array not sorted correctly\n", method.c_str());
            exit(EXIT_FAILURE);
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
    } else if (method == "shellsort2N") {
        cudaSorter = sortShell2N;
        bufferCount = 2;
    } else if (method == "shellsortKernel") {
        deviceMemory = arraySize * sizeof(int);
        bufferCount = 1;
    } else if (method == "thrustsortKernel") {
        deviceMemory = 3 * arraySize * sizeof(int);
        bufferCount = 2;
    } else {
        std::cerr << "Invalid sorting method.\n";
        exit(EXIT_FAILURE);
    }
    chunkSize = (deviceMemory / bufferCount) / sizeof(int);
    if (method.find("thrust") == 0) chunkSize = chunkSize * 0.80;
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
    method = (argc > 1) ? argv[1] : "thrustsort2N";
    distribution = (argc > 2) ? argv[2] : "uniform";
    arraySize = (argc > 3) ? std::atoi(argv[3]) : 100'000'000;
    deviceMemory = (argc > 4) ? std::atoi(argv[4]) : 500;

    seed = (argc > 5) ? std::atoi(argv[5]) : 42;
    warmup = (argc > 6) ? std::atoi(argv[6]) : 1;
    iterations = (argc > 7) ? std::atoi(argv[7]) : 3;

    std::string label = "Benchmark - Method: " + method + ", Distribution: " + distribution + 
                        ", Array Size: " + std::to_string(arraySize) + 
                        ", Array Byte Size: " + std::to_string(arraySize * sizeof(int) / (1024 * 1024)) + " MB" +
                        ", Device Memory: " + std::to_string(deviceMemory) + " MB" +
                        ", Warmup: " + std::to_string(warmup) + ", Iterations: " + std::to_string(iterations);
    printf(label.c_str());

    benchmark();
    return 0;
}
