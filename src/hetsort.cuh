#pragma once
#include <iostream>
#include <parallel/algorithm>

#include "array.cuh"
#include "kernel.cuh"

struct GPUInfo {
    int id;
    int *buffer1, *buffer2;
    bool doubleBuffer, useFirstBuffer;
    cudaStream_t stream1, stream2, streamTmp;
    size_t freeMem, totalMem, bufferSize1, bufferSize2;

    GPUInfo(int id, size_t freeMem, size_t totalMem, bool doubleBuffer);
    ~GPUInfo();
    void toggleBuffer();
    bool ensureCapacity(size_t requiredSize);
};

std::vector<GPUInfo> getGPUsInfo(bool doubleBuffer);
void splitArray(int* unsortedArray, size_t arraySize, std::vector<GPUInfo>& gpus, std::vector<std::vector<int>>& chunks, bool doubleBuffer);
void sortChunks(std::vector<std::vector<int>>& chunks, std::vector<GPUInfo>& gpus, size_t block_size = 1024 * 1024);
std::vector<int> multiWayMerge(std::vector<std::vector<int>>& chunks);
void InplaceMemcpy(int* htod_source, int* dtoh_source, int* dtoh_dest, size_t num_bytes_htod, size_t num_bytes_dtoh,
                   cudaStream_t htod_stream, cudaStream_t dtoh_stream, size_t block_size);

#define CHECK_CUDA_ERROR(err)                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                           \
    }
