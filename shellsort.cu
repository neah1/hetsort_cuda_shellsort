#include <stdio.h>

__global__ void parallelShellSort(int *d_array, int array_size, int gap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx * gap; i < array_size; i += stride * gap)
    {
        int temp = d_array[i];
        int j = i;
        for (; j >= gap && d_array[j - gap] > temp; j -= gap)
        {
            d_array[j] = d_array[j - gap];
        }
        d_array[j] = temp;
    }
}

int main()
{
    int h_array[] = {34, 7, 23, 32, 5, 62, 19, 39};
    int array_size = sizeof(h_array) / sizeof(h_array[0]);
    int *d_array;
    cudaMalloc(&d_array, array_size * sizeof(int));
    // cudaMemcpy(d_array, h_array, array_size * sizeof(int), cudaMemcpyHostToDevice);

    // int gap = 4; // Starting with a gap of 4 for demonstration
    // int threadsPerBlock = 256;
    // int blocks = (array_size + threadsPerBlock - 1) / threadsPerBlock;

    // parallelShellSort<<<blocks, threadsPerBlock>>>(d_array, array_size, gap);
    // cudaDeviceSynchronize();

    // cudaMemcpy(h_array, d_array, array_size * sizeof(int), cudaMemcpyDeviceToHost);

    // printf("Sorted Array: ");
    // for (int i = 0; i < array_size; i++)
    // {
    //     printf("%d ", h_array[i]);
    // }
    // printf("\n");

    // cudaFree(d_array);
    return 0;
}
