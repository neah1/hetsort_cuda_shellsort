#pragma once
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <time.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <unordered_map>

void generateRandomArray(int* array, size_t arraySize, int seed, std::string distribution);
bool fileExists(size_t arraySize, std::string& distribution);
void writeArrayToFile(int* array, size_t arraySize, std::string& distribution);
void readArrayFromFile(int* array, size_t arraySize, std::string& distribution);
std::unordered_map<int, int> countElements(const int* array, size_t arraySize);
bool checkArraySorted(const int* sorted, std::unordered_map<int, int> counts, size_t arraySize);