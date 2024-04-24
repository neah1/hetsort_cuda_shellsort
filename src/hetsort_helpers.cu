#include "hetsort.cuh"

GPUInfo::GPUInfo(int id, size_t bufferSize, size_t bufferCount)
        : id(id), bufferSize(bufferSize), bufferCount(bufferCount), useFirstBuffer(true) {}

void GPUInfo::initialize() {
    cudaSetDevice(id);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&streamTmp);
    cudaMalloc(&buffer1, bufferSize);
    if (bufferCount > 1) cudaMalloc(&buffer2, bufferSize);
    if (bufferCount > 2) cudaMalloc(&bufferTmp, bufferSize);
}

void GPUInfo::destroy() {
    cudaSetDevice(id);
    if (stream1 != nullptr) cudaStreamDestroy(stream1);
    if (stream2 != nullptr) cudaStreamDestroy(stream2);
    if (streamTmp != nullptr) cudaStreamDestroy(streamTmp);
    if (buffer1 != nullptr) cudaFree(buffer1);
    if (buffer2 != nullptr) cudaFree(buffer2);
    if (bufferTmp != nullptr) cudaFree(bufferTmp);
}

void GPUInfo::toggleBuffer() {
    useFirstBuffer = !useFirstBuffer;
}

std::vector<GPUInfo> getGPUsInfo(size_t deviceMemory, size_t bufferCount, size_t gpuCount) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::vector<GPUInfo> gpus;
    gpus.reserve(numGPUs);
    size_t bufferSize = deviceMemory / bufferCount;

    #pragma omp parallel for num_threads(numGPUs)
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        #pragma omp critical
        {
            if (deviceMemory <= freeMem && gpus.size() < gpuCount) {
                gpus.emplace_back(i, bufferSize, bufferCount);
                std::cout << "GPU " << i << ": " << freeMem / (1024 * 1024) << " MB free, " << totalMem / (1024 * 1024) << " MB total\n";
            } else {
                std::cout << "GPU " << i << ": " << freeMem / (1024 * 1024) << " MB free, " << totalMem / (1024 * 1024) << " MB total - Skipped\n";
            }
        }

    }
    std::cout << "GPUs available: " << gpus.size() << "\n";
    return gpus;
}

template<typename T>
T largestPowerOfTwo(T x) {
    if (x == 0) return 0;
    int bits = sizeof(T) * CHAR_BIT;
    for (int shift = 1; shift < bits; shift *= 2) x |= (x >> shift);
    return x - (x >> 1);
}

size_t nextPowerOfTwo(size_t n) {
    if (n == 0) return 1;
    if ((n & (n - 1)) == 0) return n;
    return static_cast<size_t>(pow(2, ceil(log2(n))));
}

void padVectorToPowerOfTwo(std::vector<int>& array) {
    size_t currentSize = array.size();
    size_t newSize = nextPowerOfTwo(currentSize);
    if (newSize > currentSize) {
        array.resize(newSize, INT_MAX);
        printf("Padded array from %lu to %lu\n", currentSize, newSize);
    }
}

std::vector<std::vector<std::vector<int>>> splitArray(int* unsortedArray, size_t arraySize, size_t chunkSize, std::vector<GPUInfo>& gpus, bool bitonicChunks) {
    std::vector<std::vector<int>> chunks;
    if (bitonicChunks) chunkSize = largestPowerOfTwo(chunkSize);
    size_t numChunks = arraySize / chunkSize + (arraySize % chunkSize != 0);
    std::cout << "Number of chunks: " << numChunks << "\n";
    
    // Split the array into chunks
    chunks.reserve(numChunks);
    for (size_t i = 0; i < numChunks; ++i) {
        size_t startIdx = i * chunkSize;
        size_t endIdx = std::min(startIdx + chunkSize, arraySize);
        chunks.emplace_back(unsortedArray + startIdx, unsortedArray + endIdx);
    }

    // Assign chunks to GPUs
    std::vector<std::vector<std::vector<int>>> chunkGroups(gpus.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        size_t gpuIndex = i % gpus.size();
        chunkGroups[gpuIndex].push_back(chunks[i]);
    }

    return chunkGroups;
}

std::vector<int> multiWayMerge(const std::vector<std::vector<std::vector<int>>>& chunkGroups, size_t arraySize) {
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
    merged_result.resize(arraySize);
    return merged_result;
}
