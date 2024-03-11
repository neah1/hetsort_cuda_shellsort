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

bool checkArraySorted(const int* sorted, std::unordered_map<int, int> counts, size_t size) {
    // Check for the same elements and count in the sorted array
    for (int i = 0; i < size; ++i) {
        // If any element's count goes below zero, arrays are not identical
        if (--counts[sorted[i]] < 0) {
            printf("Element %d has count less than zero.\n", sorted[i]);
            return false;
        }
        // If any element is smaller than the previous, it's not sorted
        if (i > 0 && sorted[i] < sorted[i - 1]) {
            printf("Element %d is smaller than the previous.\n", sorted[i]);
            return false;
        }
    }
    // If all elements' counts are exactly zero, arrays are identical
    for (const auto& count : counts) {
        if (count.second != 0) {
            printf("Element %d has count %d.\n", count.first, count.second);
            return false;
        }
    }

    return true;
}

bool checkChunksSorted(const std::unordered_map<int, int> counts, const std::vector<std::vector<int>>& chunks) {
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
    if (counts.size() != chunkCounts.size()) {
        printf("Mismatch in number of unique elements. Original: %zu, Aggregated: %zu\n", counts.size(), chunkCounts.size());
        sorted = false;
    }

    for (const auto& pair : counts) {
        if (chunkCounts.find(pair.first) == chunkCounts.end() || chunkCounts[pair.first] != pair.second) {
            printf("Mismatch in counts for element %d. Original: %d, Aggregated: %d\n", pair.first, pair.second, chunkCounts[pair.first]);
            sorted = false;
        }
    }

    return sorted;
}