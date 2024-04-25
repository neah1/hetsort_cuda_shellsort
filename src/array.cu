#include "array.cuh"

void generateRandomArray(int* array, size_t arraySize, int seed, std::string& distribution) {
    std::mt19937 gen(seed); // Mersenne Twister PRNG
    if (distribution == "uniform") {
        std::uniform_int_distribution<int> dis(0, INT_MAX);
        for (size_t i = 0; i < arraySize; ++i) {
            array[i] = dis(gen);
        }
    } else if (distribution == "normal") {
        std::normal_distribution<double> dis(500, 200); // mean = 500, standard deviation = 200
        for (size_t i = 0; i < arraySize; ++i) {
            array[i] = static_cast<int>(dis(gen));
        }
    } else if (distribution == "sorted") {
        std::uniform_int_distribution<int> dis(0, INT_MAX);
        for (size_t i = 0; i < arraySize; ++i) {
            array[i] = dis(gen);
        }
        std::sort(array, array + arraySize);
    } else if (distribution == "reverse_sorted") {
        std::uniform_int_distribution<int> dis(0, INT_MAX);
        for (size_t i = 0; i < arraySize; ++i) {
            array[i] = dis(gen);
        }
        std::sort(array, array + arraySize, std::greater<int>());
    } else if (distribution == "nearly_sorted") {
        std::uniform_int_distribution<int> dis(0, INT_MAX);
        for (size_t i = 0; i < arraySize; ++i) {
            array[i] = dis(gen);
        }
        std::sort(array, array + arraySize);
        // Introduce a small percentage of random swaps
        int numSwaps = arraySize * 0.01; // 1% of array size
        for (int i = 0; i < numSwaps; ++i) {
            int idx1 = std::uniform_int_distribution<int>(0, arraySize - 1)(gen);
            int idx2 = std::uniform_int_distribution<int>(0, arraySize - 1)(gen);
            std::swap(array[idx1], array[idx2]);
        }
    } else {
        printf("Invalid distribution type: %s\n", distribution.c_str());
        exit(1);
    }
}

bool fileExists(size_t arraySize, std::string& distribution) {
    std::string filename = "./data/array_" + distribution + "_" + std::to_string(arraySize) + ".bin";
    std::ifstream infile(filename);
    return infile.good();
}

void writeArrayToFile(int* array, size_t arraySize, std::string& distribution) {
    printf("Writing array to file\n");
    std::string filename = "./data/array_" + distribution + "_" + std::to_string(arraySize) + ".bin";
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<char*>(array), arraySize * sizeof(int));
    out.close();
}

void readArrayFromFile(int* array, size_t arraySize, std::string& distribution) {
    printf("Reading array from file\n");
    std::string filename = "./data/array_" + distribution + "_" + std::to_string(arraySize) + ".bin";
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(array), arraySize * sizeof(int));
    in.close();
}

std::unordered_map<int, int> countElements(const int* array, size_t arraySize) {
    std::unordered_map<int, int> counts;
    for (size_t i = 0; i < arraySize; ++i) counts[array[i]]++;
    return counts;
}

bool checkArraySorted(const int* sorted, std::unordered_map<int, int> counts, size_t arraySize) {
    bool sortedCorrectly = true;
    #pragma omp parallel for shared(sortedCorrectly)
    for (int i = 1; i < arraySize; ++i) {
        if (sortedCorrectly && sorted[i] < sorted[i - 1]) {
            #pragma omp critical
            {
                sortedCorrectly = false;
            }
        }
    }

    if (sortedCorrectly) {
        #pragma omp parallel
        {
            std::unordered_map<int, int> local_counts;
            #pragma omp for nowait
            for (int i = 0; i < arraySize; ++i) {
                local_counts[sorted[i]]++;
            }
            #pragma omp critical
            {
                for (const auto &pair : local_counts) {
                    counts[pair.first] -= pair.second;
                }
            }
        }

        std::vector<std::pair<const int, int>> elements(counts.begin(), counts.end());
        #pragma omp parallel for shared(sortedCorrectly)
        for (size_t i = 0; i < elements.size(); ++i) {
            if (sortedCorrectly && elements[i].second != 0) {
                #pragma omp critical
                {
                    sortedCorrectly = false;
                }
            }
        }
    }

    if (sortedCorrectly) {
        printf("Array is sorted correctly\n");
    } else {
        fprintf(stderr, "Error: Array not sorted correctly\n");
    }
    return sortedCorrectly;
}