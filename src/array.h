#ifndef ARRAY_H
#define ARRAY_H
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <cuda_runtime.h>

void printArray(const int* array, int size);
void generateRandomArray(int* array, int size, int seed);
bool checkArraySorted(const int* original, const int* sorted, int size);

#define CHECK_CUDA_ERROR(err)                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                           \
    }

#endif