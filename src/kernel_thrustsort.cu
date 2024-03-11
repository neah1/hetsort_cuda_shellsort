#include "kernel.cuh"

void thrustsort(int* d_array, size_t arraySize, cudaStream_t stream) {
    thrust::device_ptr<int> array_ptr(d_array);
    thrust::sort(thrust::cuda::par.on(stream), array_ptr, array_ptr + arraySize);
}
