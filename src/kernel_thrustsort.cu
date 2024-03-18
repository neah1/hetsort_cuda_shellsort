#include "kernel.cuh"

class custom_temporary_allocator {
public:
    using value_type = int;

private:
    int* buffer_start;
    size_t buffer_size;
    int* current_ptr;

public:
    custom_temporary_allocator(int* buffer_start, size_t buffer_size)
        : buffer_start(buffer_start), buffer_size(buffer_size), current_ptr(buffer_start) {
    }

    int* allocate(std::ptrdiff_t num_bytes) {
        if (static_cast<size_t>(current_ptr - buffer_start + num_bytes) > buffer_size)
            throw std::bad_alloc();

        int* allocation_start = current_ptr;
        current_ptr += num_bytes;
        return allocation_start;
    }

    void deallocate(int* ptr, size_t n) {}
};

void thrustsort(int* d_array, size_t arraySize, int* buffer, size_t bufferSize, cudaStream_t stream) {
    // Create device pointer for the array to be sorted
    thrust::device_ptr<int> array_ptr(d_array);

    // Create an instance of the custom allocator with the auxiliary buffer
    custom_temporary_allocator allocator(buffer, bufferSize);

    // Create a temporary dynamic memory allocation policy from the custom allocator
    auto alloc_policy = thrust::cuda::par(allocator).on(stream);

    // Perform the sort operation using the specified stream and auxiliary buffer for temporary storage
    thrust::sort(alloc_policy, array_ptr, array_ptr + arraySize);
}