#include "hetsort.cuh"

GPUInfo::GPUInfo(int id, size_t freeMem, size_t totalMem, bool doubleBuffer)
    : id(id), freeMem(freeMem), totalMem(totalMem), doubleBuffer(doubleBuffer), buffer1(nullptr), buffer2(nullptr), bufferSize1(0), bufferSize2(0), useFirstBuffer(true) {
    cudaSetDevice(id);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&streamTmp);
    if (doubleBuffer) cudaStreamCreate(&stream2);
}

GPUInfo::~GPUInfo() {
    cudaSetDevice(id);
    cudaFree(buffer1);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(streamTmp);
    if (doubleBuffer) cudaFree(buffer2);
    if (doubleBuffer) cudaStreamDestroy(stream2);
}

void GPUInfo::toggleBuffer() {
    if (doubleBuffer) useFirstBuffer = !useFirstBuffer;
}

bool GPUInfo::ensureCapacity(size_t requiredSize) {
    if (freeMem < 500 * 1024 * 1024) return false;

    size_t& bufferSize = useFirstBuffer ? bufferSize1 : bufferSize2;
    size_t requiredMem = requiredSize - bufferSize;
    if (freeMem < requiredMem) return false;

    cudaSetDevice(id);
    if (requiredSize <= bufferSize) return true;

    int** buffer = useFirstBuffer ? &buffer1 : &buffer2;
    cudaFree(*buffer);
    cudaError_t err = cudaMalloc((void**)buffer, requiredSize);
    if (err != cudaSuccess) return false;

    freeMem -= requiredMem;
    bufferSize1 = requiredSize;
    return true;
}

std::vector<GPUInfo> getGPUsInfo(bool doubleBuffer) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::vector<GPUInfo> gpus;
    gpus.reserve(numGPUs);

    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        gpus.emplace_back(i, freeMem, totalMem, doubleBuffer);
        printf("GPU %d: %zu MB free, %zu MB total\n", i, freeMem / (1024 * 1024), totalMem / (1024 * 1024));
    }
    return gpus;
}

void splitArray(int* unsortedArray, size_t arraySize, std::vector<GPUInfo>& gpus, std::vector<std::vector<int>>& chunks, bool doubleBuffer) {
    // Calculate the chunk size based on the array size and the number of GPUs
    size_t chunkSize = arraySize / gpus.size();
    if (doubleBuffer) chunkSize /= 2;

    // Determine the number of chunks needed
    size_t numChunks = arraySize / chunkSize + (arraySize % chunkSize != 0);
    chunks.reserve(numChunks);

    // Split the array into chunks
    for (size_t i = 0; i < numChunks; ++i) {
        size_t startIdx = i * chunkSize;
        size_t endIdx = std::min(startIdx + chunkSize, arraySize);
        chunks.emplace_back(unsortedArray + startIdx, unsortedArray + endIdx);
    }
}

std::vector<int> multiWayMerge(std::vector<std::vector<int>>& chunks) {
    // Prepare sequences for the multi-way merge
    std::vector<std::pair<int*, int*>> sequences(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        sequences[i] = std::make_pair(chunks[i].data(), chunks[i].data() + chunks[i].size());
    }

    // Allocate memory for the final merged result
    size_t total_size = 0;
    for (const auto& chunk : chunks) total_size += chunk.size();
    std::vector<int> merged_result(total_size);

    // Perform the multiway merge
    __gnu_parallel::multiway_merge(sequences.begin(), sequences.end(), merged_result.begin(), total_size, std::less<int>());
    return merged_result;
}

int main(int argc, char* argv[]) {
    int seed = 42;
    size_t arraySize = (argc > 1) ? std::atoi(argv[1]) : 1'000'000;
    bool doubleBuffer = (argc > 2) ? std::atoi(argv[2]) : false;
    printf("Array size is set to: %zu. Double buffer: %s\n", arraySize, doubleBuffer ? "true" : "false");

    // Allocate and initialize arrays
    size_t arrayByteSize = arraySize * sizeof(int);
    int* h_inputArray = (int*)malloc(arrayByteSize);
    generateRandomArray(h_inputArray, arraySize, seed);
    std::unordered_map<int, int> counts = countElements(h_inputArray, arraySize);

    // Get GPU information
    std::vector<GPUInfo> gpus = getGPUsInfo(doubleBuffer);

    // Split the array into chunks based on GPU memory availability
    std::vector<std::vector<int>> chunks;
    splitArray(h_inputArray, arraySize, gpus, chunks, doubleBuffer);

    // Sort each chunk on the GPU
    sortChunks(chunks, gpus);

    // Check if each chunk is sorted correctly
    if (checkChunksSorted(counts, chunks)) printf("Chunks are sorted correctly\n");

    // Perform multi-way merge
    std::vector<int> merged_result = multiWayMerge(chunks);

    // Check if the merged array is sorted correctly
    if (checkArraySorted(merged_result.data(), counts, arraySize)) printf("Array is sorted correctly\n");

    // Clean up
    free(h_inputArray);
    return 0;
}