#include <iostream>
#include <parallel/algorithm>
#include "array.h"
#include "shellsort.h"
#include "inplace_memcpy.cuh"

struct GPUInfo {
    int id;
    int *buffer1, *buffer2;
    bool doubleBuffer, useFirstBuffer;
    cudaStream_t stream1, stream2, streamTmp;
    size_t freeMem, totalMem, bufferSize1, bufferSize2;

    GPUInfo(int id, size_t freeMem, size_t totalMem, bool doubleBuffer)
        : id(id), freeMem(freeMem), totalMem(totalMem), doubleBuffer(doubleBuffer), 
        buffer1(nullptr), buffer2(nullptr), bufferSize1(0), bufferSize2(0), useFirstBuffer(true) {
        cudaSetDevice(id);
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&streamTmp);
        if (doubleBuffer) cudaStreamCreate(&stream2);
    }

    ~GPUInfo() {
        cudaSetDevice(id);
        cudaFree(buffer1);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(streamTmp);
        if (doubleBuffer) cudaFree(buffer2);
        if (doubleBuffer) cudaStreamDestroy(stream2);
    }

    void toggleBuffer() {
        if (doubleBuffer) useFirstBuffer = !useFirstBuffer;
    }

    bool ensureCapacity(size_t requiredSize) {
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
};

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

void splitArrayIntoChunks(int* unsortedArray, size_t arraySize, std::vector<GPUInfo>& gpus, std::vector<std::vector<int>>& chunks, bool doubleBuffer) {
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

void sortChunks(std::vector<std::vector<int>>& chunks, std::vector<GPUInfo>& gpus, size_t block_size = 1024 * 1024) {
    size_t lastGPU = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        size_t chunkByteSize = chunks[i].size() * sizeof(int);

        for (size_t j = 0; j < gpus.size(); ++j) {
            int gpuId = (lastGPU + j) % gpus.size();
            GPUInfo& gpu = gpus[gpuId];

            if (!gpu.ensureCapacity(chunkByteSize)) {
                printf("Chunk %d (%zu MB) is too large for GPU %d (%zu MB)\n", i, chunkByteSize / (1024 * 1024), gpu.id, gpu.freeMem / (1024 * 1024));
                continue;
            } else {
                printf("Sorting Chunk %d (%zu MB) on GPU %d (%zu MB)\n", i, chunkByteSize / (1024 * 1024), gpu.id, gpu.freeMem / (1024 * 1024));
            }

            int* currentBuffer = gpu.useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
            cudaStream_t& currentStream = gpu.useFirstBuffer ? gpu.stream1 : gpu.stream2;
            size_t currentBufferSize = gpu.useFirstBuffer ? gpu.bufferSize1 : gpu.bufferSize2;

            thrustsort(currentBuffer, chunks[i].size(), currentStream);
            
            lastGPU = gpuId + 1;
            gpu.toggleBuffer();
            break;
        }
    }

    for (auto& gpu : gpus) {
        cudaStreamSynchronize(gpu.stream1);
        if (gpu.doubleBuffer) cudaStreamSynchronize(gpu.stream2);
    }
}

std::vector<int> multiWayMerge(std::vector<std::vector<int>>& chunks) {
    // Prepare sequences for the multi-way merge
    std::vector<std::pair<int*, int*>> sequences(chunks.size());
    for(size_t i = 0; i < chunks.size(); ++i) {
        sequences[i] = std::make_pair(chunks[i].data(), chunks[i].data() + chunks[i].size());
    }

    // Allocate memory for the final merged result
    size_t total_size = 0;
    for(const auto& chunk : chunks) total_size += chunk.size();
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
    splitArrayIntoChunks(h_inputArray, arraySize, gpus, chunks, doubleBuffer);

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