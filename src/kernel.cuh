#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/memory.h>
#include <nvtx3/nvToolsExt.h>

void shellsort(int* d_array, size_t arraySize, cudaStream_t stream);
void thrustsort(int* d_array, size_t arraySize, int* buffer, size_t bufferSize, cudaStream_t stream);