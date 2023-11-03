#include <cuda.h>
#include <cuda_runtime.h>

__global__ void parallelShellSort(int *array, int arraySize, int increment)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index >= increment)
        return;

    for (int i = index; i < arraySize; i += increment)
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