#include "hetsort.cuh"

void sortThrust2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t block_size) {
    omp_set_num_threads(gpus.size());
    #pragma omp parallel for
    for (int g = 0; g < static_cast<int>(chunkGroups.size()); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        cudaSetDevice(gpu.id);

        for (size_t i = 0; i < chunks.size(); ++i) {
            int* mainBuffer = gpu.useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
            int* secondaryBuffer = gpu.useFirstBuffer ? gpu.buffer2 : gpu.buffer1;

            // Copy the first chunk data to the main GPU buffer
            size_t mainChunkSize = chunks[i].size() * sizeof(int);
            if (i == 0) cudaMemcpyAsync(mainBuffer, chunks[i].data(), mainChunkSize, cudaMemcpyHostToDevice, gpu.stream1);

            // Sort the chunk
            thrustsort(mainBuffer, chunks[i].size(), secondaryBuffer, gpu.bufferSize, gpu.stream1);

            // Copy the sorted chunk back, and the next chunk to the other GPU buffer
            cudaStreamSynchronize(gpu.stream1);
            cudaMemcpyAsync(chunks[i].data(), mainBuffer, mainChunkSize, cudaMemcpyDeviceToHost, gpu.stream1);
            if (i + 1 < chunks.size()) {
                cudaMemcpyAsync(secondaryBuffer, chunks[i + 1].data(), chunks[i + 1].size() * sizeof(int), cudaMemcpyHostToDevice, gpu.streamTmp);
                cudaStreamSynchronize(gpu.streamTmp);
            }

             // Toggle the buffer for the next chunk
            gpu.toggleBuffer();
        }
    }
    for (auto& gpu : gpus) cudaStreamSynchronize(gpu.stream1);
}

void sortThrustInplace(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t block_size) {}
void sortShell(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t block_size) {}
void sortShell2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus, size_t block_size) {}