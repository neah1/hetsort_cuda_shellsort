#include "hetsort.cuh"

void sortChunks(std::vector<std::vector<int>>& chunks, std::vector<GPUInfo>& gpus, size_t block_size) {
    size_t lastGPU = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        size_t chunkByteSize = chunks[i].size() * sizeof(int);

        for (size_t j = 0; j < gpus.size(); ++j) {
            int gpuId = (lastGPU + j) % gpus.size();
            GPUInfo& gpu = gpus[gpuId];

            if (!gpu.ensureCapacity(chunkByteSize)) {
                printf("Chunk %d (%zu MB) is too large for GPU %d (%zu MB)\n", i, chunkByteSize / (1024 * 1024), gpu.id, gpu.freeMem / (1024 * 1024));
                continue;
            } else {
                printf("Sorting Chunk %d (%zu MB) on GPU %d (%zu MB)\n", i, chunkByteSize / (1024 * 1024), gpu.id, gpu.freeMem / (1024 * 1024));
            }

            int* currentBuffer = gpu.useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
            cudaStream_t& currentStream = gpu.useFirstBuffer ? gpu.stream1 : gpu.stream2;
            size_t currentBufferSize = gpu.useFirstBuffer ? gpu.bufferSize1 : gpu.bufferSize2;

            thrustsort(currentBuffer, chunks[i].size(), currentStream);

            lastGPU = gpuId + 1;
            gpu.toggleBuffer();
            break;
        }
    }

    for (auto& gpu : gpus) {
        cudaStreamSynchronize(gpu.stream1);
        if (gpu.doubleBuffer) cudaStreamSynchronize(gpu.stream2);
    }
}