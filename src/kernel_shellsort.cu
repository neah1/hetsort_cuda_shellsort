#include "kernel.cuh"

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

__device__ void bitonicCompare(int* values, int i, int j, bool dir) {
    int temp;
    if ((values[i] > values[j]) == dir) {
        temp = values[i];
        values[i] = values[j];
        values[j] = temp;
    }
}

__global__ void bitonicSortKernel(int* d_array, size_t arraySize) {
    extern __shared__ int shared[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory. Pad with maximum value if this is the last segment and it's not full
    shared[tid] = (idx < arraySize) ? d_array[idx] : INT_MAX;
    __syncthreads();

    // Bitonic sort in shared memory
    for (int size = 2; size <= blockDim.x; size *= 2) {
        bool dir = ((tid / size) % 2 == 0);
        for (int stride = size / 2; stride > 0; stride /= 2) {
            int pos = (tid / size) * size + (tid % size);
            bitonicCompare(shared, pos, pos + stride, dir);
            __syncthreads();
        }
    }
    // Write sorted elements back to global memory
    if (idx < arraySize) d_array[idx] = shared[tid];
}

__device__ void simpleMerge(int* shared, size_t segmentSize) {
    size_t index1 = threadIdx.x;
    size_t index2 = segmentSize + index1;

    if (index1 < segmentSize) {
        if (shared[index1] > shared[index2]) {
            int temp = shared[index1];
            shared[index1] = shared[index2];
            shared[index2] = temp;
        }
    }
}

__global__ void mergesortKernel(int* d_array, size_t arraySize, size_t segmentSize) {
    extern __shared__ int shared[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory, padding if necessary
    if (idx < arraySize) {
        shared[tid] = d_array[idx];
    } else {
        shared[tid] = INT_MAX;
    }
    __syncthreads();

    // Perform simple in-place merge
    simpleMerge(shared, segmentSize);
    __syncthreads();

    // Write merged segment back to global memory
    if (idx < arraySize) d_array[idx] = shared[tid];
}

void mergesort(int* d_array, size_t arraySize, size_t initialSegmentSize, cudaStream_t stream) {
    size_t segmentSize = initialSegmentSize;
    size_t numSegments = (arraySize + segmentSize - 1) / segmentSize;

    while (numSegments > 1) {
        size_t mergedSegmentSize = segmentSize * 2; // Size of the segment after merging
        size_t numBlocks = (arraySize + mergedSegmentSize - 1) / mergedSegmentSize; // One block per merged segment
        size_t sharedMemorySize = mergedSegmentSize * sizeof(int); // Allocate enough shared memory for merged segment

        mergesortKernel<<<numBlocks, segmentSize, sharedMemorySize, stream>>>(d_array, arraySize, segmentSize);

        cudaStreamSynchronize(stream); // Ensure all merges are complete before next iteration

        segmentSize = mergedSegmentSize; // Update segment size for next iteration
        numSegments = (arraySize + segmentSize - 1) / segmentSize; // Update number of segments
    }
}

void shellsort(int* d_array, size_t arraySize, cudaStream_t stream) {
    const size_t MAX_THREADS_PER_BLOCK = 1024;
    const size_t BITONIC_SORT_THRESHOLD = 2048;
    size_t numThreads, numBlocks, blockSize;

    std::vector<size_t> increments = generateIncrements(arraySize);

    for (size_t increment : increments) {
        if (increment >= BITONIC_SORT_THRESHOLD) {
            numThreads = min(MAX_THREADS_PER_BLOCK, increment);
            numBlocks = (increment + numThreads - 1) / numThreads;
            shellsortKernel<<<numBlocks, numThreads, 0, stream>>>(d_array, arraySize, increment);
        } else {
            blockSize = BITONIC_SORT_THRESHOLD;
            numBlocks = (arraySize + blockSize - 1) / blockSize;
            size_t sharedMemSize = blockSize * sizeof(int);
            bitonicSortKernel<<<numBlocks, blockSize, sharedMemSize, stream>>>(d_array, arraySize);
            break;
        }
    }

    mergesort(d_array, arraySize, blockSize, stream);
}