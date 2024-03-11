#ifndef ARRAY_H
#define ARRAY_H
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

void printArray(const int* array, size_t size);
void generateRandomArray(int* array, size_t size, int seed);
std::unordered_map<int, int> countElements(const int* array, size_t size);
bool checkArraySorted(const int* sorted, std::unordered_map<int, int> counts, size_t size);
bool checkChunksSorted(const std::unordered_map<int, int> originalCounts, const std::vector<std::vector<int>>& chunks);

#define CHECK_CUDA_ERROR(err)                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                           \
    }

#endif