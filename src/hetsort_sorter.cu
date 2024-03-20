#include "hetsort.cuh"

void sortThrust2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus) {
    omp_set_num_threads(gpus.size());
    #pragma omp parallel for
    for (int g = 0; g < static_cast<int>(chunkGroups.size()); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        cudaSetDevice(gpu.id);

        for (size_t i = 0; i < chunks.size(); ++i) {
            int* mainBuffer = gpu.useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
            int* secondaryBuffer = gpu.useFirstBuffer ? gpu.buffer2 : gpu.buffer1;

            if (i == 0) doubleMemcpy(mainBuffer, chunks[i].data(), chunks[i].size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);

            thrustsort(mainBuffer, chunks[i].size(), secondaryBuffer, gpu.bufferSize, gpu.stream1);

            if (i + 1 < chunks.size()) {
                cudaStreamSynchronize(gpu.stream1);
                cudaMemcpyAsync(chunks[i].data(), mainBuffer, chunks[i].size() * sizeof(int), cudaMemcpyDeviceToHost, gpu.stream1);
                cudaMemcpyAsync(secondaryBuffer, chunks[i + 1].data(), chunks[i + 1].size() * sizeof(int), cudaMemcpyHostToDevice, gpu.streamTmp);
                cudaStreamSynchronize(gpu.streamTmp);
            } else {
                doubleMemcpy(chunks[i].data(), mainBuffer, chunks[i].size(), cudaMemcpyDeviceToHost, gpu.stream1, gpu.streamTmp);
            }

            gpu.toggleBuffer();
        }
    }
    for (auto& gpu : gpus) cudaStreamSynchronize(gpu.stream1);
}

void sortThrustInplace(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus) {
    omp_set_num_threads(gpus.size());
    #pragma omp parallel for
    for (int g = 0; g < static_cast<int>(chunkGroups.size()); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        cudaSetDevice(gpu.id);

        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i == 0) doubleMemcpy(gpu.buffer1, chunks[i].data(), chunks[i].size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);

            thrustsort(gpu.buffer1, chunks[i].size(), gpu.buffer2, gpu.bufferSize, gpu.stream1);

            doubleMemcpy(chunks[i].data(), gpu.buffer1, chunks[i].size(), cudaMemcpyDeviceToHost, gpu.stream1, gpu.streamTmp);
            if (i + 1 < chunks.size()) doubleMemcpy(gpu.buffer1, chunks[i + 1].data(), chunks[i + 1].size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);

            // int* nextChunkData = (i + 1 < chunks.size()) ? chunks[i + 1].data() : nullptr;
            // size_t nextChunkSize = (i + 1 < chunks.size()) ? chunks[i + 1].size() * sizeof(int) : 0;
            // InplaceMemcpy(nextChunkData, gpu.buffer1, chunks[i].data(), nextChunkSize, chunks[i].size() * sizeof(int), gpu.stream1, gpu.streamTmp, blockSize);
        }
    }
    for (auto& gpu : gpus) cudaStreamSynchronize(gpu.stream1);
}

void sortShell(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus) {
    omp_set_num_threads(gpus.size());
    #pragma omp parallel for
    for (int g = 0; g < static_cast<int>(chunkGroups.size()); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        cudaSetDevice(gpu.id);

        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i == 0) doubleMemcpy(gpu.buffer1, chunks[i].data(), chunks[i].size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);

            shellsort(gpu.buffer1, chunks[i].size(), gpu.stream1);

            doubleMemcpy(chunks[i].data(), gpu.buffer1, chunks[i].size(), cudaMemcpyDeviceToHost, gpu.stream1, gpu.streamTmp);
            if (i + 1 < chunks.size()) doubleMemcpy(gpu.buffer1, chunks[i + 1].data(), chunks[i + 1].size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);

            // int* nextChunkData = (i + 1 < chunks.size()) ? chunks[i + 1].data() : nullptr;
            // size_t nextChunkSize = (i + 1 < chunks.size()) ? chunks[i + 1].size() * sizeof(int) : 0;
            // InplaceMemcpy(nextChunkData, gpu.buffer1, chunks[i].data(), nextChunkSize, chunks[i].size() * sizeof(int), gpu.stream1, gpu.streamTmp, blockSize);
        }
    }
    for (auto& gpu : gpus) cudaStreamSynchronize(gpu.stream1);
}

void sortShell2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus) {
    omp_set_num_threads(gpus.size());
    #pragma omp parallel for
    for (int g = 0; g < static_cast<int>(chunkGroups.size()); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        cudaSetDevice(gpu.id);

        for (size_t i = 0; i < chunks.size(); ++i) {
            int* mainBuffer = gpu.useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
            int* secondaryBuffer = gpu.useFirstBuffer ? gpu.buffer2 : gpu.buffer1;
            cudaStream_t mainStream = gpu.useFirstBuffer ? gpu.stream1 : gpu.stream2;
            cudaStream_t secondaryStream = gpu.useFirstBuffer ? gpu.stream2 : gpu.stream1;

            // Copy the first chunk data to the main GPU buffer
            if (i == 0) doubleMemcpy(mainBuffer, chunks[i].data(), chunks[i].size(), cudaMemcpyHostToDevice, mainStream, gpu.streamTmp);

            // Sort the chunk on the main buffer
            shellsort(mainBuffer, chunks[i].size(), mainStream);

            // Copy the sorted chunk back and the next chunk to the secondary buffer
            if ((i > 0)) {
                doubleMemcpy(chunks[i - 1].data(), secondaryBuffer, chunks[i - 1].size(), cudaMemcpyDeviceToHost, secondaryStream, gpu.streamTmp);
            }
            if (i + 1 < chunks.size()) {
                doubleMemcpy(secondaryBuffer, chunks[i + 1].data(), chunks[i + 1].size(), cudaMemcpyHostToDevice, secondaryStream, gpu.streamTmp);
            }

            // Copy the sorted chunk back and the next chunk to the secondary buffer
            // int* nextChunkData = (i + 1 < chunks.size()) ? chunks[i + 1].data() : nullptr;
            // size_t nextChunkSize = (i + 1 < chunks.size()) ? chunks[i + 1].size() * sizeof(int) : 0;
            // int* prevChunkData = (i > 0) ? chunks[i - 1].data() : nullptr;
            // size_t prevChunkSize = (i > 0) ? chunks[i - 1].size() * sizeof(int) : 0;
            // InplaceMemcpy(nextChunkData, secondaryBuffer, prevChunkData, nextChunkSize, prevChunkSize, secondaryStream, gpu.streamTmp, blockSize);

            // Copy the last sorted chunk back
            if (i == chunks.size() - 1) cudaMemcpyAsync(chunks[i].data(), mainBuffer, chunks[i].size() * sizeof(int), cudaMemcpyDeviceToHost, mainStream);

            // Toggle the buffer for the next chunk
            gpu.toggleBuffer();
        }
    }
    for (auto& gpu : gpus) {
        cudaStreamSynchronize(gpu.stream1);
        cudaStreamSynchronize(gpu.stream2);
    }
}