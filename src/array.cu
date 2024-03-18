#include "array.cuh"

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

bool checkChunkGroupsSorted(const std::unordered_map<int, int>& originalCounts, const std::vector<std::vector<std::vector<int>>>& chunkGroups) {
    std::unordered_map<int, int> chunkGroupCounts;
    bool sorted = true;

    for (size_t g = 0; g < chunkGroups.size(); ++g) {
        const auto& chunks = chunkGroups[g];
        for (size_t i = 0; i < chunks.size(); ++i) {
            const auto& chunk = chunks[i];
            for (int num : chunk) chunkGroupCounts[num]++;

            if (!std::is_sorted(chunk.begin(), chunk.end())) {
                printf("Chunk %zu in group %zu is not sorted.\n", i, g);
                sorted = false;
            }
        }
    }

    // Compare aggregated chunk group counts to original counts
    if (originalCounts.size() != chunkGroupCounts.size()) {
        printf("Mismatch in number of unique elements. Original: %zu, Aggregated: %zu\n", originalCounts.size(), chunkGroupCounts.size());
        sorted = false;
    }
    
    for (const auto& pair : originalCounts) {
        if (chunkGroupCounts.find(pair.first) == chunkGroupCounts.end() || chunkGroupCounts[pair.first] != pair.second) {
            printf("Mismatch in counts for element %d. Original: %d, Aggregated: %d\n", pair.first, pair.second, chunkGroupCounts[pair.first]);
            sorted = false;
        }
    }

    return sorted;
}
