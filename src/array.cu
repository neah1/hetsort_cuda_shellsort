#include "array.cuh"

void generateRandomArray(int* array, size_t arraySize, int seed) {
    srand(seed);
    for (int i = 0; i < arraySize; ++i) array[i] = rand() % 1000;
}

void printArray(const int* array, size_t arraySize) {
    printf("[");
    for (int i = 0; i < arraySize; ++i) {
        printf("%d", array[i]);
        if (i < arraySize - 1) printf(", ");
    }
    printf("]\n");
}

std::unordered_map<int, int> countElements(const int* array, size_t arraySize) {
    std::unordered_map<int, int> counts;
    for (size_t i = 0; i < arraySize; ++i) counts[array[i]]++;
    return counts;
}

bool checkChunkGroupsSorted(const std::vector<std::vector<std::vector<int>>>& chunkGroups, const std::unordered_map<int, int>& counts) {
    std::unordered_map<int, int> chunkGroupCounts;

    for (size_t g = 0; g < chunkGroups.size(); ++g) {
        const auto& chunks = chunkGroups[g];
        for (size_t i = 0; i < chunks.size(); ++i) {
            const auto& chunk = chunks[i];
            for (int num : chunk) chunkGroupCounts[num]++;
            if (!std::is_sorted(chunk.begin(), chunk.end())) {
                printf("Chunk %zu in group %zu is not sorted.\n", i, g);
                return false;
            }
        }
    }

    if (counts.size() != chunkGroupCounts.size()) {
        printf("Mismatch in number of unique elements. Original: %zu, Sorted: %zu\n", counts.size(), chunkGroupCounts.size());
        return false;
    }
    
    for (const auto& pair : counts) {
        if (chunkGroupCounts.find(pair.first) == chunkGroupCounts.end() || chunkGroupCounts[pair.first] != pair.second) {
            printf("Mismatch in counts for element %d. Original: %d, Sorted: %d\n", pair.first, pair.second, chunkGroupCounts[pair.first]);
            return false;
        }
    }

    return true;
}

bool checkArraySorted(const int* sorted, std::unordered_map<int, int> counts, size_t arraySize) {
    for (int i = 0; i < arraySize; ++i) {
        if (--counts[sorted[i]] < 0) {
            printf("Element %d has count less than zero.\n", sorted[i]);
            return false;
        }
    
        if (i > 0 && sorted[i] < sorted[i - 1]) {
            printf("Element %d is smaller than the previous.\n", sorted[i]);
            return false;
        }
    }
    for (const auto& count : counts) {
        if (count.second != 0) {
            printf("Element %d has count more than zero.\n", count.first);
            return false;
        }
    }

    return true;
}