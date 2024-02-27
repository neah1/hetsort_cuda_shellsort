#include <iostream>
#include <parallel/algorithm>
#include "array.h"
#include "shellsort.h"
#include "inplace_memcpy.cuh"

struct GPUInfo {
    int id;
    size_t freeMem, totalMem, bufferSize;
    int *buffer1, *buffer2; // Reusable device memory
    cudaStream_t stream1, stream2, stream3; // CUDA stream for asynchronous operations

    GPUInfo(int id, size_t freeMem, size_t totalMem)
        : id(id), freeMem(freeMem), totalMem(totalMem), bufferSize(0), buffer1(nullptr), buffer2(nullptr) {
        cudaSetDevice(id);
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);
    }

    ~GPUInfo() {
        cudaSetDevice(id);
        cudaFree(buffer1); // Free the allocated memory
        cudaFree(buffer2); // Free the second buffer
        cudaStreamDestroy(stream1); // Destroy the stream
        cudaStreamDestroy(stream2); // Destroy the second stream
        cudaStreamDestroy(stream3); // Destroy the third stream
    }

    // Ensure the device array has enough memory allocated for the chunk
    bool ensureCapacity(size_t requiredSize, bool doubleBuffer = false) {
        // TODO: Handle this case
        if (requiredSize * 2 > freeMem) return false; 
        if (freeMem / (1024 * 1024) < 300) return false;

        if (doubleBuffer) requiredSize /= 2;
        if (requiredSize > bufferSize) {
            bufferSize = requiredSize;
            cudaFree(buffer1); // Free the old device memory
            cudaMalloc(&buffer1, requiredSize); // Allocate new memory
            if (doubleBuffer) {
                cudaFree(buffer2); // Free the old device memory
                cudaMalloc(&buffer2, requiredSize); // Allocate new memory
            }
        }
        return true;
    }
};

std::vector<GPUInfo> getGPUsInfo() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::vector<GPUInfo> gpus;
    gpus.reserve(numGPUs);

    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        gpus.emplace_back(i, freeMem, totalMem);
        printf("GPU %d: %zu MB free, %zu MB total\n", i, freeMem / (1024 * 1024), totalMem / (1024 * 1024));
    }
    return gpus;
}

void splitArrayIntoChunks(int* unsortedArray, size_t arraySize, std::vector<GPUInfo>& gpus, std::vector<std::vector<int>>& chunks, bool doubleBuffer = false) {
    // // Calculate chunk size based on available memory across all GPUs
    // size_t totalGPUMem = 0;
    // for (const auto& gpu : gpus) totalGPUMem += gpu.freeMem;

    // // Estimate chunk size based on total GPU memory, leaving ~10MB margin per GPU
    // size_t totalAvailableMem = totalGPUMem - gpus.size() * (10 * 1024 * 1024);
    // size_t chunkSize = totalAvailableMem / gpus.size() / sizeof(int) / 2; // Thrust sort

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

void sortChunks(std::vector<std::vector<int>>& chunks, std::vector<GPUInfo>& gpus) {
    // Pre-allocate based on the first chunk as a simple heuristic
    for (auto& gpu : gpus) {
        cudaSetDevice(gpu.id);
        gpu.ensureCapacity(chunks[0].size() * sizeof(int));
    }

    size_t lastGPU = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        size_t chunkByteSize = chunks[i].size() * sizeof(int);

        for (size_t j = 0; j < gpus.size(); ++j) {
            int gpuId = (lastGPU + j) % gpus.size();
            GPUInfo& gpu = gpus[gpuId];
            cudaSetDevice(gpu.id);

            if (!gpu.ensureCapacity(chunkByteSize)) {
                printf("Chunk %d (%zu MB) is too large for GPU %d (%zu MB)\n", i, chunkByteSize / (1024 * 1024), gpu.id, gpu.freeMem / (1024 * 1024));
                continue;
            }
            printf("Sorting Chunk %d (%zu MB) on GPU %d (%zu MB)\n", i, chunkByteSize / (1024 * 1024), gpu.id, gpu.freeMem / (1024 * 1024));

            cudaMemcpyAsync(gpu.buffer1, chunks[i].data(), chunkByteSize, cudaMemcpyHostToDevice, gpu.stream1);
            thrustsort(gpu.buffer1, chunks[i].size(), gpu.stream1);
            cudaMemcpyAsync(chunks[i].data(), gpu.buffer1, chunkByteSize, cudaMemcpyDeviceToHost, gpu.stream1);

            lastGPU = gpuId + 1;
            break;
        }
    }

    for (auto& gpu : gpus) cudaStreamSynchronize(gpu.stream1);
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
    printf("Array size is set to: %zu\n", arraySize);

    // Allocate and initialize arrays
    size_t arrayByteSize = arraySize * sizeof(int);
    int* h_inputArray = (int*)malloc(arrayByteSize);
    generateRandomArray(h_inputArray, arraySize, seed);
    std::unordered_map<int, int> counts = countElements(h_inputArray, arraySize);

    // Get GPU information
    std::vector<GPUInfo> gpus = getGPUsInfo();

    // Split the array into chunks based on GPU memory availability
    std::vector<std::vector<int>> chunks;
    splitArrayIntoChunks(h_inputArray, arraySize, gpus, chunks);

    // Sort each chunk on the GPU
    sortChunks(chunks, gpus);

    // Check if each chunk is sorted correctly
    if (checkChunksSorted(counts, chunks)) printf("Chunks are sorted correctly\n");

    // Perform multi-way merge
    std::vector<int> merged_result = multiWayMerge(chunks);

    if (checkArraySorted(merged_result.data(), counts, arraySize)) printf("Array is sorted correctly\n");

    // Clean up
    free(h_inputArray);
    return 0;
}

// void sortChunks2N(std::vector<std::vector<int>>& chunks, std::vector<GPUInfo>& gpus) {
//     size_t chunkByteSize = chunks[0].size() * sizeof(int);

//     // Prepare streams and buffers for each GPU
//     for (auto& gpu : gpus) {
//         cudaSetDevice(gpu.id);
//         gpu.ensureCapacity(chunkByteSize / 2, true); // Allocate half of the total memory for each buffer
//     }

//     // Distribute chunks round-robin and manage double buffering
//     size_t gpuId = 0;
//     for (size_t i = 0; i < chunks.size(); ++i) {
//         // Alternate between buffer1 and buffer2 for each chunk
//         bool useFirstBuffer = (i / gpus.size()) % 2 == 0;
//         GPUInfo& gpu = gpus[gpuId];
//         cudaSetDevice(gpu.id);

//         int* currentBuffer = useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
//         cudaStream_t& currentStream = useFirstBuffer ? gpu.stream1 : gpu.stream2;

//         // Asynchronous copy to GPU
//         cudaMemcpyAsync(currentBuffer, chunks[i].data(), chunkByteSize, cudaMemcpyHostToDevice, currentStream);
        
//         // Launch sorting operation asynchronously
//         thrustsort(currentBuffer, chunks[i].size(), currentStream);

//         // Asynchronous copy back to host
//         cudaMemcpyAsync(chunks[i].data(), currentBuffer, chunkByteSize, cudaMemcpyDeviceToHost, currentStream);

//         gpuId = (gpuId + 1) % gpus.size(); // Move to the next GPU
//     }

//     // Synchronize all GPUs
//     for (auto& gpu : gpus) {
//         cudaStreamSynchronize(gpu.stream1);
//         cudaStreamSynchronize(gpu.stream2);
//     }
// }

// InplaceMemcpy(chunks[i + 1].data(), nullptr, chunks[i-1].data(), chunkByteSize, chunkByteSize, gpu.stream1, gpu.stream3, chunkByteSize);