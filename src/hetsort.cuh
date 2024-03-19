#pragma once
#include <omp.h>
#include <parallel/algorithm>
#include "array.cuh"
#include "kernel.cuh"

struct GPUInfo {
    int id;
    bool useFirstBuffer;
    size_t bufferSize, bufferCount;
    int *buffer1, *buffer2, *bufferTmp;
    cudaStream_t stream1, stream2, streamTmp;

    GPUInfo(int id, size_t bufferSize, size_t bufferCount);
    ~GPUInfo();
    void toggleBuffer();
};

std::vector<GPUInfo> getGPUsInfo(size_t deviceMemory, size_t bufferCount);
std::vector<std::vector<std::vector<int>>> splitArray(int* unsortedArray, size_t arraySize, size_t chunkSize, std::vector<GPUInfo>& gpus);
std::vector<int> multiWayMerge(const std::vector<std::vector<std::vector<int>>>& chunkGroups);

void sortShell(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t blockSize);
void sortShell2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t blockSize);
void sortThrust2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t blockSize);
void sortThrustInplace(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t blockSize);

void doubleMemcpy(int* d_array, const int* h_array, size_t arraySize, cudaStream_t stream1, cudaStream_t stream2);
void InplaceMemcpy(int* htod_source, int* dtoh_source, int* dtoh_dest, size_t num_bytes_htod, size_t num_bytes_dtoh,
                   cudaStream_t htod_stream, cudaStream_t dtoh_stream, size_t block_size);

#define CHECK_CUDA_ERROR(err)                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                           \
    }
