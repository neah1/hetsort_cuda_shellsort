#include "hetsort.cuh"

GPUInfo::GPUInfo(int id, size_t bufferSize, bool buffers2N)
    : id(id), bufferSize(bufferSize), buffers2N(buffers2N), useFirstBuffer(true) {
    cudaSetDevice(id);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&streamTmp);
    cudaMalloc(&buffer1, bufferSize);
    if (buffers2N) {
        cudaStreamCreate(&stream2);
        cudaMalloc(&buffer2, bufferSize);
    }
}

GPUInfo::~GPUInfo() {
    cudaSetDevice(id);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(streamTmp);
    cudaFree(buffer1);
    if (buffers2N) {
        cudaStreamDestroy(stream2);
        cudaFree(buffer2);
    }
}

void GPUInfo::toggleBuffer() {
    if (buffers2N) useFirstBuffer = !useFirstBuffer;
}

std::vector<GPUInfo> getGPUsInfo(size_t bufferSize, bool buffers2N) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::vector<GPUInfo> gpus;
    gpus.reserve(numGPUs);
    size_t requiredMem = buffers2N ? bufferSize * 2 : bufferSize;

    #pragma omp parallel for
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        #pragma omp critical
        {
            if (freeMem >= requiredMem) {
                gpus.emplace_back(i, bufferSize, buffers2N);
                printf("GPU %d: %zu MB free, %zu MB total\n", i, freeMem / (1024 * 1024), totalMem / (1024 * 1024));
            } else {
                printf("GPU %d: %zu MB free, %zu MB total - Skipped\n", i, freeMem / (1024 * 1024), totalMem / (1024 * 1024));
            }
        }

    }
    printf("GPUs available: %zu\n", gpus.size());
    return gpus;
}

std::vector<std::vector<std::vector<int>>> splitArray(int* unsortedArray, size_t arraySize, size_t bufferSize, std::vector<GPUInfo>& gpus) {
    std::vector<std::vector<int>> chunks;
    
    size_t numChunks = arraySize / bufferSize + (arraySize % bufferSize != 0);
    chunks.reserve(numChunks);

    // Split the array into chunks
    for (size_t i = 0; i < numChunks; ++i) {
        size_t startIdx = i * bufferSize;
        size_t endIdx = std::min(startIdx + bufferSize, arraySize);
        chunks.emplace_back(unsortedArray + startIdx, unsortedArray + endIdx);
    }

    size_t numGPUs = gpus.size();
    std::vector<std::vector<std::vector<int>>> chunkGroups(numGPUs);

    // Assign chunks to GPUs
    for (size_t i = 0; i < chunks.size(); ++i) {
        size_t gpuIndex = i % numGPUs;
        chunkGroups[gpuIndex].push_back(chunks[i]);
    }

    return chunkGroups;
}

std::vector<int> multiWayMerge(const std::vector<std::vector<std::vector<int>>>& chunkGroups) {
    // Prepare a vector of sequences for the multi-way merge from chunk groups
    std::vector<std::pair<int*, int*>> sequences;
    for (const auto& group : chunkGroups) {
        for (const auto& chunk : group) {
            if (!chunk.empty()) sequences.emplace_back(const_cast<int*>(chunk.data()), const_cast<int*>(chunk.data()) + chunk.size());
        }
    }

    // Calculate the total size for the merged result
    size_t total_size = 0;
    for (const auto& seq : sequences) total_size += std::distance(seq.first, seq.second);
    std::vector<int> merged_result(total_size);

    // Perform the multiway merge
    __gnu_parallel::multiway_merge(sequences.begin(), sequences.end(), merged_result.begin(), total_size, std::less<int>());
    return merged_result;
}

int main(int argc, char* argv[]) {
    int seed = 42;
    size_t arraySize = (argc > 1) ? std::atoi(argv[1]) : 1'000'000;
    size_t bufferSize = (argc > 2) ? std::atoi(argv[2]) : 200;
    bool buffers2N = (argc > 3) ? std::atoi(argv[3]) : true;
    printf("Array size: %zu. Buffer size: %zu MB. Double buffer: %s\n", arraySize, bufferSize, buffers2N ? "true" : "false");

    // Allocate and initialize arrays
    int* h_inputArray = (int*)malloc(arraySize * sizeof(int));
    generateRandomArray(h_inputArray, arraySize, seed);
    std::unordered_map<int, int> originalCounts = countElements(h_inputArray, arraySize);

    // Get GPU information
    bufferSize = bufferSize * 1024 * 1024;
    std::vector<GPUInfo> gpus = getGPUsInfo(bufferSize, buffers2N);

    // Split the array into chunks based on GPU memory availability
    std::vector<std::vector<std::vector<int>>> chunkGroups = splitArray(h_inputArray, arraySize, bufferSize, gpus);

    // Sort each chunk on the GPU
    sortChunkGroups(chunkGroups, gpus);

    // Check if each chunk is sorted correctly
    if (checkChunkGroupsSorted(originalCounts, chunkGroups)) printf("Chunks are sorted correctly\n");

    // Perform multi-way merge
    std::vector<int> merged_result = multiWayMerge(chunkGroups);

    // Check if the merged array is sorted correctly
    if (checkArraySorted(merged_result.data(), originalCounts, arraySize)) printf("Array is sorted correctly\n");

    // Clean up
    free(h_inputArray);
    return 0;
}