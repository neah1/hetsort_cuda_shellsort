#pragma once
#include <omp.h>
#include <iostream>
#include <parallel/algorithm>

#include "array.cuh"
#include "kernel.cuh"

struct GPUInfo {
    int id;
    size_t bufferSize;
    int *buffer1, *buffer2;
    bool buffers2N, useFirstBuffer;
    cudaStream_t stream1, stream2, streamTmp;

    GPUInfo(int id, size_t bufferSize, bool buffers2N);
    ~GPUInfo();
    void toggleBuffer();
};

std::vector<GPUInfo> getGPUsInfo(size_t bufferSize, bool buffers2N);
std::vector<std::vector<std::vector<int>>> splitArray(int* unsortedArray, size_t arraySize, size_t bufferSize, std::vector<GPUInfo>& gpus);
std::vector<int> multiWayMerge(const std::vector<std::vector<std::vector<int>>>& chunkGroups);

void sortShell(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t block_size = 1024 * 1024);
void sortShell2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t block_size = 1024 * 1024);
void sortThrust2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t block_size = 1024 * 1024);
void sortThrustInplace(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t block_size = 1024 * 1024);

void InplaceMemcpy(int* htod_source, int* dtoh_source, int* dtoh_dest, size_t num_bytes_htod, size_t num_bytes_dtoh,
                   cudaStream_t htod_stream, cudaStream_t dtoh_stream, size_t block_size);

#define CHECK_CUDA_ERROR(err)                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                           \
    }
