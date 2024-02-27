#include "array.h"

void generateRandomArray(int* array, size_t size, int seed) {
    srand(seed);
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % 1000;
    }
}

void printArray(const int* array, size_t size) {
    printf("[");
    for (int i = 0; i < size; ++i) {
        printf("%d", array[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

std::unordered_map<int, int> countElements(const int* array, size_t size) {
    std::unordered_map<int, int> counts;
    for (size_t i = 0; i < size; ++i) counts[array[i]]++;
    return counts;
}

bool checkArraySorted(const int* original, const int* sorted, size_t size) {
    std::unordered_map<int, int> counts = countElements(original, size);

    // Check for the same elements and count in the sorted array
    for (int i = 0; i < size; ++i) {
        // If any element's count goes below zero, arrays are not identical
        if (--counts[sorted[i]] < 0) return false;
        // If any element is smaller than the previous, it's not sorted
        if (i > 0 && sorted[i] < sorted[i - 1]) return false;
    }

    // If all elements' counts are exactly zero, arrays are identical
    for (const auto &count : counts) {
        if (count.second != 0) return false;
    }

    return true;
}

bool checkChunksSorted(const std::unordered_map<int, int> originalCounts, const std::vector<std::vector<int>>& chunks) {
    std::unordered_map<int, int> chunkCounts;
    bool sorted = true;

    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        for (int num : chunk) chunkCounts[num]++;

        if (!std::is_sorted(chunk.begin(), chunk.end())) {
            printf("Chunk %zu is not sorted.\n", i);
            sorted = false;
        }
    }

    // Compare aggregated chunk counts to original counts
    if (originalCounts.size() != chunkCounts.size()) {
        printf("Mismatch in number of unique elements. Original: %zu, Aggregated: %zu\n", originalCounts.size(), chunkCounts.size());
        sorted = false;
    }

    for (const auto& pair : originalCounts) {
        if (chunkCounts.find(pair.first) == chunkCounts.end() || chunkCounts[pair.first] != pair.second) {
            printf("Mismatch in counts for element %d. Original: %d, Aggregated: %d\n", pair.first, pair.second, chunkCounts[pair.first]);
            sorted = false;
        }
    }

    return sorted;
}