#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <thrust/sort.h>
#include "shellsort.h"
#include "array.h"


struct GPUInfo {
    int id;
    int *buffer1, *buffer2; // Reusable device memory
    size_t freeMem, totalMem, bufferSize;
    cudaStream_t stream1, stream2; // CUDA stream for asynchronous operations

    GPUInfo(int id, size_t freeMem, size_t totalMem)
        : id(id), freeMem(freeMem), totalMem(totalMem), buffer1(nullptr), buffer2(nullptr), bufferSize(0) {
        cudaSetDevice(id);
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
    }

    ~GPUInfo() {
        cudaSetDevice(id);
        cudaFree(buffer1); // Free the allocated memory
        cudaFree(buffer2); // Free the second buffer
        cudaStreamDestroy(stream1); // Destroy the stream
        cudaStreamDestroy(stream2); // Destroy the second stream
    }

    // Ensure the device array has enough memory allocated for the chunk
    void ensureCapacity(size_t requiredSize, bool doubleBuffer = false) {
        if (requiredSize > bufferSize) {
            cudaFree(buffer1); // Free the old device memory
            cudaMalloc(&buffer1, requiredSize); // Allocate new memory
            bufferSize = requiredSize;
            if (doubleBuffer) {
                cudaFree(buffer2); // Free the old device memory
                cudaMalloc(&buffer2, requiredSize); // Allocate new memory
                bufferSize = requiredSize;
            }
        }
    }
};

std::vector<GPUInfo> getGPUsInfo() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::vector<GPUInfo> gpus(numGPUs);

    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        gpus[i] = {i, freeMem, totalMem};
        std::cout << "GPU " << i << ": " << freeMem / (1024 * 1024) << " MB free, "
                  << totalMem / (1024 * 1024) << " MB total" << std::endl;
    }
    return gpus;
}

void splitArrayIntoChunks(int* unsortedArray, size_t arraySize, std::vector<GPUInfo>& gpus, std::vector<std::vector<int>>& chunks, bool doubleBuffer = false) {
    // Calculate chunk size based on available memory across all GPUs
    size_t totalGPUMem = 0;
    for (const auto& gpu : gpus) {
        totalGPUMem += gpu.freeMem;
    }

    // Estimate chunk size based on total GPU memory, leaving ~10MB margin per GPU
    size_t totalAvailableMem = totalGPUMem - gpus.size() * (10 * 1024 * 1024);
    size_t chunkSize = totalAvailableMem / sizeof(int) / gpus.size();
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

void distributeAndSortChunks(int* unsortedArray, size_t arraySize, std::vector<GPUInfo>& gpus, std::vector<std::vector<int>> chunks) {
    size_t chunkByteSize = chunks[0].size() * sizeof(int);

    for (auto& gpu : gpus) {
        cudaSetDevice(gpu.id);
        gpu.ensureCapacity(chunkByteSize);
    }


    for (size_t i = 0; i < chunks.size(); ++i) {
        int gpuId = i % gpus.size();
        GPUInfo& gpu = gpus[gpuId];
        cudaSetDevice(gpu.id);

        cudaMemcpyAsync(gpu.buffer1, chunks[i].data(), chunkByteSize, cudaMemcpyHostToDevice, gpu.stream1);

        thrustsort(gpu.buffer1, chunks[i].size(), gpu.stream1);

        cudaMemcpyAsync(chunks[i].data(), gpu.buffer1, chunkByteSize, cudaMemcpyDeviceToHost, gpu.stream1);
    }

    for (auto& gpu : gpus) cudaStreamSynchronize(gpu.stream1);
}

void distributeAndSortChunks2N(std::vector<std::vector<int>>& chunks, std::vector<GPUInfo>& gpus) {
    size_t chunkByteSize = chunks[0].size() * sizeof(int);

    // Prepare streams and buffers for each GPU
    for (auto& gpu : gpus) {
        cudaSetDevice(gpu.id);
        gpu.ensureCapacity(chunkByteSize / 2, true); // Allocate half of the total memory for each buffer
    }

    // Distribute chunks round-robin and manage double buffering
    size_t gpuId = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        // Alternate between buffer1 and buffer2 for each chunk
        bool useFirstBuffer = (i / gpus.size()) % 2 == 0;
        GPUInfo& gpu = gpus[gpuId];
        cudaSetDevice(gpu.id);

        int* currentBuffer = useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
        cudaStream_t& currentStream = useFirstBuffer ? gpu.stream1 : gpu.stream2;

        // Asynchronous copy to GPU
        cudaMemcpyAsync(currentBuffer, chunks[i].data(), chunkByteSize, cudaMemcpyHostToDevice, currentStream);
        
        // Launch sorting operation asynchronously
        thrustsort(currentBuffer, chunks[i].size(), currentStream);

        // Asynchronous copy back to host
        cudaMemcpyAsync(chunks[i].data(), currentBuffer, chunkByteSize, cudaMemcpyDeviceToHost, currentStream);

        gpuId = (gpuId + 1) % gpus.size(); // Move to the next GPU
    }

    // Synchronize all GPUs
    for (auto& gpu : gpus) {
        cudaStreamSynchronize(gpu.stream1);
        cudaStreamSynchronize(gpu.stream2);
    }
}



// TODO Final merge

