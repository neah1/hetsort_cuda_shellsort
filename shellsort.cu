#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(err)                                         \
    if (err != cudaSuccess)                                           \
    {                                                                 \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                           \
    }

__global__ void parallelShellSort(int *array, int arraySize, int increment)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < arraySize; i += stride)
    {
        int temp = array[i];
        int j;
        for (j = i; j >= increment && array[j - increment] > temp; j -= increment)
        {
            array[j] = array[j - increment];
        }
        array[j] = temp;
    }
}

int main()
{
    // TEMPORARY INPUT ARRAY
    int h_inputArray[] = {34, 7, 23, 32, 5, 62, 19, 39};
    int arraySize = sizeof(h_inputArray) / sizeof(h_inputArray[0]);
    size_t arrayByteSize = arraySize * sizeof(int);

    // Allocate device memory
    int *d_inputArray;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_inputArray, arrayByteSize));
    CHECK_CUDA_ERROR(cudaMemcpy(d_inputArray, h_inputArray, arrayByteSize, cudaMemcpyHostToDevice));

    int increments[] = {121, 40, 13, 4, 1}; // Example increment sequence from Ciura (2001)
    int numThreads = 1;
    int numBlocks = (arraySize + numThreads - 1) / numThreads;
    for (int i = 0; i < sizeof(increments) / sizeof(increments[0]); i++)
    {
        int increment = increments[i];
        parallelShellSort<<<numBlocks, numThreads>>>(d_inputArray, arraySize, increment);
        cudaDeviceSynchronize(); // Ensure kernel execution is finished before next iteration
    }

    // Copy sorted array back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_inputArray, d_inputArray, arrayByteSize, cudaMemcpyDeviceToHost));

    // Print sorted array
    for (int i = 0; i < arraySize; i++)
    {
        printf("%d ", h_inputArray[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_inputArray);

    return 0;
}
