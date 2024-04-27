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
    void initialize();
    void destroy();
    void toggleBuffer();
};

std::vector<GPUInfo> getGPUsInfo(size_t deviceMemory, size_t bufferCount, size_t gpuCount);
size_t nextPowerOfTwo(size_t n);
void padVectorToPowerOfTwo(std::vector<int>& data);
std::vector<std::vector<std::vector<int>>> splitArray(int* unsortedArray, size_t arraySize, size_t chunkSize, std::vector<GPUInfo>& gpus, bool bitonicChunks);

std::vector<int> multiWayMerge(const std::vector<std::vector<std::vector<int>>>& chunkGroups, size_t arraySize);
std::vector<int> sortKernel(const std::string& method, int* h_inputArray, size_t arraySize, std::vector<GPUInfo>& gpus);

void sortShell(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus);
void sortShell2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus);
void sortThrust2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus);
void sortThrust3N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus);
void sortThrustInplace(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus);
void sortThrustInplaceMemcpy(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus);

void doubleMemcpy(int* dest_array, const int* source_array, size_t arraySize, cudaMemcpyKind memcpyMode, cudaStream_t stream1, cudaStream_t stream2);
void InplaceMemcpy(int* htod_source, int* dtoh_source, int* dtoh_dest, size_t num_bytes_htod, size_t num_bytes_dtoh, cudaStream_t htod_stream, cudaStream_t dtoh_stream);

#define CHECK_CUDA_ERROR(err)                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                           \
    }
