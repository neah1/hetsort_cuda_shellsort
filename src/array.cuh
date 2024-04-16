#pragma once
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <random>

void generateRandomArray(int* array, size_t arraySize, int seed, std::string distribution);
std::unordered_map<int, int> countElements(const int* array, size_t arraySize);
bool checkArraySorted(const int* sorted, std::unordered_map<int, int> counts, size_t arraySize);
bool checkChunkGroupsSorted(const std::vector<std::vector<std::vector<int>>>& chunkGroups, const std::unordered_map<int, int>& counts);