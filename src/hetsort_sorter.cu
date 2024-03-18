#include "hetsort.cuh"

void sortChunkGroups(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t block_size) {
    // Make sure the number of threads matches the number of GPUs/chunk groups
    omp_set_num_threads(gpus.size());

    #pragma omp parallel for
    for (int g = 0; g < static_cast<int>(chunkGroups.size()); ++g) {
        GPUInfo& gpu = gpus[g];
        auto& chunks = chunkGroups[g];

        for (size_t i = 0; i < chunks.size(); ++i) {
            size_t chunkByteSize = chunks[i].size() * sizeof(int);

            // int* currentBuffer = gpu.useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
            // cudaStream_t& currentStream = gpu.useFirstBuffer ? gpu.stream1 : gpu.stream2;

            // Make sure CUDA calls use the correct GPU for this thread
            cudaSetDevice(gpu.id);

            // Copy the chunk data to the GPU buffer
            cudaMemcpyAsync(gpu.buffer1, chunks[i].data(), chunkByteSize, cudaMemcpyHostToDevice, gpu.stream1);
            // Sort the chunk
            thrustsort(gpu.buffer1, chunks[i].size(), gpu.buffer2, gpu.bufferSize, gpu.stream1);
            // Copy the sorted chunk back
            cudaMemcpyAsync(chunks[i].data(), gpu.buffer1, chunkByteSize, cudaMemcpyDeviceToHost, gpu.stream1);

            gpu.toggleBuffer(); // Toggle the buffer for the next chunk if double buffering
        }

        // cudaStreamSynchronize(gpu.stream1);
        // if (gpu.buffers2N) cudaStreamSynchronize(gpu.stream2);
    }

    for (auto& gpu : gpus) {
        cudaStreamSynchronize(gpu.stream1);
        // if (gpu.buffers2N) cudaStreamSynchronize(gpu.stream2);
    }
}

// void sortThrust2N(std::vector<std::vector<int>>& chunks, std::vector<GPUInfo>& gpus, size_t block_size) {}