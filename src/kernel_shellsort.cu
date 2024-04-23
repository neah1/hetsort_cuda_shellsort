#include "kernel.cuh"

__global__ void shellsortKernel(int* d_array, size_t arraySize, size_t increment) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    int temp, j;

    if (index >= increment) return;
    for (size_t i = index; i < arraySize; i += increment) {
        temp = d_array[i];
        for (j = i; j >= increment && d_array[j - increment] > temp; j -= increment) {
            d_array[j] = d_array[j - increment];
        }
        d_array[j] = temp;
    }
}

__device__ void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

__global__ void bitonicKernel(int *d_array, size_t arraySize, int k, int j) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = tid ^ j;

    if (ixj > tid) {
        if ((tid & k) == 0) {
            if (d_array[tid] > d_array[ixj]) {
				swap(d_array[tid], d_array[ixj]);
            }
        }
        else {
            if (d_array[tid] < d_array[ixj]) {
				swap(d_array[tid], d_array[ixj]);
            }
        }
    }
}

void bitonicSort(int* d_array, size_t arraySize, cudaStream_t stream) {
    int threadsPerBlock = 256;
    int numBlocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    for (int k = 2; k <= arraySize; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_array, arraySize, k, j);
            cudaStreamSynchronize(stream);
        }
    }
}

std::vector<size_t> generateIncrements(size_t arraySize) {
    std::vector<size_t> increments = {1, 4, 10, 23, 57, 132, 301, 701, 1750};
    const double factor = 2.2; // Growth factor for the sequence

    // Calculate the maximum number of increments needed based on O(log N)
    size_t maxIncrements = static_cast<size_t>(std::log2(arraySize));

    // Generate increments larger than 1750, and then insert 2048 as a ^2 increment
    size_t lastIncrement = 1750;
    increments.push_back(2048);
    while (increments.size() < maxIncrements) {
        size_t nextIncrement = static_cast<size_t>(lastIncrement * factor);
        increments.push_back(nextIncrement);
        lastIncrement = nextIncrement;
    }

    // Reverse the sequence to start from the largest increment
    std::reverse(increments.begin(), increments.end());
    return increments;
}

void shellsort(int* d_array, size_t arraySize, cudaStream_t stream) {
    const size_t MAX_THREADS_PER_BLOCK = 1024;
    const size_t BITONIC_SORT_THRESHOLD = 2048;
    size_t numThreads, numBlocks;

    std::vector<size_t> increments = generateIncrements(arraySize);

    for (size_t increment : increments) {
        if (increment >= BITONIC_SORT_THRESHOLD) {
            numThreads = min(MAX_THREADS_PER_BLOCK, increment);
            numBlocks = (increment + numThreads - 1) / numThreads;
            shellsortKernel<<<numBlocks, numThreads, 0, stream>>>(d_array, arraySize, increment);
        }
    }

    bitonicSort(d_array, arraySize, stream);
}