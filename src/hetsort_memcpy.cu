#include "hetsort.cuh"

void doubleMemcpy(int* d_array, const int* h_array, size_t arraySize, cudaStream_t stream1, cudaStream_t stream2) {
    size_t halfSize = arraySize / 2;
    size_t halfSizeBytes = halfSize * sizeof(int);
    size_t arraySizeBytes = arraySize * sizeof(int);

    // Start async copy of the first half of the array
    cudaMemcpyAsync(d_array, h_array, halfSizeBytes, cudaMemcpyHostToDevice, stream1);

    // Start async copy of the second half of the array
    cudaMemcpyAsync(d_array + halfSize, h_array + halfSize, arraySizeBytes - halfSizeBytes, cudaMemcpyHostToDevice, stream2);

    // Wait for both streams to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
}

void InplaceMemcpy(int* htod_source, int* dtoh_source, int* dtoh_dest, size_t num_bytes_htod, size_t num_bytes_dtoh,
                   cudaStream_t htod_stream, cudaStream_t dtoh_stream, size_t block_size) {
    if (dtoh_dest == nullptr && htod_source == nullptr) {
        return;
    }

    size_t num_bytes;

    if (dtoh_dest == nullptr) {
        num_bytes = num_bytes_htod;
        block_size = num_bytes;
    } else if (htod_source == nullptr) {
        num_bytes = num_bytes_dtoh;
        block_size = num_bytes;
    } else {
        num_bytes = std::min(num_bytes_htod, num_bytes_dtoh);
        block_size = std::min(block_size, num_bytes);
    }

    size_t bytes_dtoh = 0;

    if (dtoh_dest != nullptr) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(dtoh_dest, dtoh_source, block_size, cudaMemcpyDeviceToHost, dtoh_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(dtoh_stream));
    }
    bytes_dtoh += block_size;

    while (bytes_dtoh < num_bytes && dtoh_dest != nullptr && htod_source != nullptr) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(dtoh_source + bytes_dtoh - block_size, htod_source + bytes_dtoh - block_size,
                                         block_size, cudaMemcpyHostToDevice, htod_stream));

        block_size = std::min(block_size, num_bytes - bytes_dtoh);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(dtoh_dest + bytes_dtoh, dtoh_source + bytes_dtoh, block_size, cudaMemcpyDeviceToHost,
                                         dtoh_stream));

        CHECK_CUDA_ERROR(cudaStreamSynchronize(htod_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(dtoh_stream));

        bytes_dtoh += block_size;
    }

    if (htod_source != nullptr) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(dtoh_source + bytes_dtoh - block_size, htod_source + bytes_dtoh - block_size,
                                         block_size, cudaMemcpyHostToDevice, htod_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(htod_stream));
    }

    if (num_bytes_htod != num_bytes_dtoh && dtoh_dest != nullptr && htod_source != nullptr) {
        if (num_bytes_htod > num_bytes_dtoh) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(dtoh_source + num_bytes, htod_source + num_bytes, num_bytes_htod - num_bytes,
                                             cudaMemcpyHostToDevice, htod_stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(htod_stream));

        } else if (num_bytes_dtoh > num_bytes_htod) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(dtoh_dest + num_bytes, dtoh_source + num_bytes, num_bytes_dtoh - num_bytes,
                                             cudaMemcpyDeviceToHost, dtoh_stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(dtoh_stream));
        }
    }
}