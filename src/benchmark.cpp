#include "benchmark.h"

void generateRandomArray(int *array, int size, int seed)
{
    srand(seed);
    for (int i = 0; i < size; ++i)
    {
        array[i] = rand() % 1000;
    }
}

void printArray(const int *array, int size)
{
    printf("[");
    for (int i = 0; i < size; ++i)
    {
        printf("%d", array[i]);
        if (i < size - 1)
        {
            printf(", ");
        }
    }
    printf("]\n");
}

bool checkArraySorted(const int *original, const int *sorted, int size)
{
    std::unordered_map<int, int> counts;
    // Count elements in the original array
    for (int i = 0; i < size; ++i)
    {
        counts[original[i]]++;
    }

    // Check for the same elements and count in the sorted array
    for (int i = 0; i < size; ++i)
    {
        if (--counts[sorted[i]] < 0)
        {
            // If any element's count goes below zero, arrays are not identical
            return false;
        }
        // Check if sorted correctly
        if (i > 0 && sorted[i] < sorted[i - 1])
        {
            // If any element is smaller than the previous, it's not sorted
            return false;
        }
    }

    // If all elements' counts are exactly zero, arrays are identical
    for (const auto &count : counts)
    {
        if (count.second != 0)
        {
            return false;
        }
    }

    return true;
}