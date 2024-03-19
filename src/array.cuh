#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

void printArray(const int* array, size_t arraySize);
void generateRandomArray(int* array, size_t arraySize, int seed);
std::unordered_map<int, int> countElements(const int* array, size_t arraySize);
bool checkArraySorted(const int* sorted, std::unordered_map<int, int> counts, size_t arraySize);
bool checkChunkGroupsSorted(const std::vector<std::vector<std::vector<int>>>& chunkGroups, const std::unordered_map<int, int>& counts);