#include "hetsort.cuh"

// Algorithm parameters
std::string method = "shellsort2N";
std::string distribution = "uniform";
size_t arraySize = 1'000'000'000;
size_t deviceMemory = 1'000;

// Algorithm parameters
const int seed = 42;
const int warmup = 0;
const int iterations = 1;

typedef void CUDASorter(std::vector<std::vector<std::vector<int>>>&, std::vector<GPUInfo>&);

std::vector<int> runSort(CUDASorter cudaSorter, int* h_inputArray, size_t arraySize, size_t chunkSize, std::vector<GPUInfo>& gpus) {
    nvtxRangePush("Split array phase");
    if (method.find("thrust") == 0) chunkSize = chunkSize * 0.80;
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

void benchmark(CUDASorter cudaSorter, int* h_inputArray, size_t arraySize, size_t chunkSize, std::vector<GPUInfo>& gpus) {
    // Count the number of elements in the input array
    std::unordered_map<int, int> originalCounts = countElements(h_inputArray, arraySize);

    // Warmup the GPU
    for (int i = 0; i < warmup; i++) runSort(cudaSorter, h_inputArray, arraySize, chunkSize, gpus);

    for (int i = 0; i < iterations; i++) {
        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        
        nvtxRangePush("HETSort algorithm");
        std::vector<int> h_outputArray = runSort(cudaSorter, h_inputArray, arraySize, chunkSize, gpus);
        nvtxRangePop();

        // Stop timing
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        printf("Iteration %d: %f ms\n", i, duration);

        // Check if the array is sorted correctly
        if (!checkArraySorted(h_outputArray.data(), originalCounts, arraySize)) {
            fprintf(stderr, "Error (%s): Array not sorted correctly\n", method.c_str());
            exit(EXIT_FAILURE);
        }
    }
}

CUDASorter* selectSortingMethod(const std::string& method, size_t& bufferCount) {
    CUDASorter* cudaSorter = nullptr;
    if (method == "thrustsort2N") {
        cudaSorter = sortThrust2N;
        bufferCount = 2;
    } else if (method == "thrustsort3N") {
        cudaSorter = sortThrust3N;
        bufferCount = 3;
    } else if (method == "thrustsortInplace") {
        cudaSorter = sortThrustInplace;
        bufferCount = 2;
    } else if (method == "shellsort") {
        cudaSorter = sortShell;
        bufferCount = 1;
    } else if (method == "shellsort2N") {
        cudaSorter = sortShell2N;
        bufferCount = 2;
    } else {
        std::cerr << "Invalid sorting method.\n";
        exit(EXIT_FAILURE);
    }
    return cudaSorter;
}

void runSortingAlgorithm(const std::string& method, const std::string& distribution, size_t arraySize, size_t deviceMemory) {
    // Select sorting method
    size_t bufferCount;
    CUDASorter* cudaSorter = selectSortingMethod(method, bufferCount);

    // Calculate chunk size
    deviceMemory = deviceMemory * 1024 * 1024;
    size_t chunkSize = (deviceMemory / bufferCount) / sizeof(int);

    nvtxRangePush("Get GPU information");
    std::vector<GPUInfo> gpus = getGPUsInfo(deviceMemory, bufferCount);
    nvtxRangePop();

    nvtxRangePush("Generate array distribution");
    int* h_inputArray;
    cudaMallocHost((void**)&h_inputArray, arraySize * sizeof(int));
    generateRandomArray(h_inputArray, arraySize, seed, distribution);
    nvtxRangePop();

    // Run sorting algorithm
    benchmark(cudaSorter, h_inputArray, arraySize, chunkSize, gpus);

    // Clean up
    cudaFreeHost(h_inputArray);
}
void fullBenchmark() {
    std::vector<std::string> methods = {"thrustsort2N", "thrustsort3N", "thrustsortInplace", "shellsort", "shellsort2N"};
    std::vector<std::string> distributions = {"uniform", "normal", "sorted", "reverse_sorted", "nearly_sorted"};
    std::vector<size_t> arraySizes = {1'000'000, 10'000'000};
    std::vector<size_t> deviceMemories = {500, 1000, 2000};

    // Loop over each combination of parameters
    for (const auto& method : methods) {
        for (const auto& distribution : distributions) {
            for (const auto& arraySize : arraySizes) {
                for (const auto& deviceMemory : deviceMemories) {
                    runSortingAlgorithm(method, distribution, arraySize, deviceMemory);
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    method = (argc > 1) ? argv[1] : method;
    distribution = (argc > 2) ? argv[2] : distribution;
    arraySize = (argc > 3) ? std::atoi(argv[3]) : arraySize;
    deviceMemory = (argc > 4) ? std::atoi(argv[4]) : deviceMemory;
    printf("Method: %s. Distribution: %s. Array size: %zu. Array byte size: %zu MB. Device memory: %zu MB. Warmup: %d. Iterations: %d.\n", 
        method.c_str(), distribution.c_str(), arraySize, arraySize * sizeof(int) / (1024 * 1024), deviceMemory, warmup, iterations);

    runSortingAlgorithm(method, distribution, arraySize, deviceMemory);
    return 0;
}