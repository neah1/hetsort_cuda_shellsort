#include <cuda.h>
#include <cuda_runtime.h>

__global__ void parallelShellsort(int *array, int arraySize, int increment) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int temp;
    int j;

    if (index >= increment) return;

    for (int i = index; i < arraySize; i += increment) {
        temp = array[i];
        for (j = i; j >= increment && array[j - increment] > temp; j -= increment) {
            array[j] = array[j - increment];
        }
        array[j] = temp;
    }
}

// __global__ void parallelShellsort(int *array, int arraySize, int increment) {
//     shellsortIncrement(array, arraySize, 1);
// }