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

__device__ void _bitonicStep1(int * smem, int tid, int tpp, int d) {
	int m = tid / (d >> 1);
	int tib = tid - m*(d >> 1);
	int addr1 = d*m + tib;
	int addr2 = (m + 1)*d - tib - 1;
	
	int A = smem[addr1];
	int B = smem[addr2];
	smem[addr1] = min(A, B);
	smem[addr2] = max(A, B);
}

__device__ void _bitonicStep2(int * smem, int tid, int tpp, int d) {
	int m = tid / (d >> 1);
	int tib = tid - m*(d >> 1);
	int addr1 = d*m + tib;
	int addr2 = addr1 + (d >> 1);

	int A = smem[addr1];
	int B = smem[addr2];
	smem[addr1] = min(A, B);
	smem[addr2] = max(A, B);
}

__global__ void bitonicKernel(int* mem) {
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	__shared__ int smem[256]; // Two blocks worth of shared memory
	smem[tpp] = mem[blockDim.x*(2 * bid) + tpp]; // Coalesced memory load
	smem[tpp + blockDim.x] = mem[blockDim.x*((2 * bid) + 1) + tpp]; // Coalesced memory load
	int blocks = 8;
	for (int blockNum = 1; blockNum <= blocks; blockNum++) {
		int d = 1 << blockNum;
		_bitonicStep1(smem, tpp, tpp, d);
		__syncthreads();
		d = d >> 1;
		while(d >= 2) {
			_bitonicStep2(smem, tpp, tpp, d);
			__syncthreads();
			d = d >> 1;
		}
	}

	mem[blockDim.x*(2 * bid) + tpp] = smem[tpp];
	mem[blockDim.x*((2*bid)+1) + tpp] = smem[tpp + blockDim.x];
}

__global__ void bitonicKernelXBlock1(int* mem, int blockNum) {
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	int d = 1 << blockNum;
	_bitonicStep1(mem, tid, tpp, d);
}

__global__ void bitonicKernelXBlock2(int* mem, int blockNum, int d) {
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	_bitonicStep2(mem, tid, tpp, d);
}

void bitonicSort(int* d_array, size_t arraySize, cudaStream_t stream) {
	// Launch a kernel on the GPU with one thread for each element.
	int numBlocks = log2(arraySize);

	bitonicKernel << <arraySize / 256, 128, 0, stream >> >(d_array);
	for (int b = 9; b <= numBlocks; b++) {
		int d = 1 << b;
		bitonicKernelXBlock1 << <arraySize / 512, 256, 0, stream >> >(d_array, b);
		d = d >> 1;
		while (d >= 2) {
			bitonicKernelXBlock2 << <arraySize / 512, 256, 0, stream >> >(d_array, b, d);
			d = d >> 1;
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

    // blockSize = BITONIC_SORT_THRESHOLD;
    // numBlocks = (arraySize + blockSize - 1) / blockSize;
    // size_t sharedMemSize = blockSize * sizeof(int);
    // bitonicKernel<<<MAX_THREADS_PER_BLOCK, numBlocks, sharedMemSize, stream>>>(d_array, arraySize);
    
    bitonicSort(d_array, arraySize, stream);
}