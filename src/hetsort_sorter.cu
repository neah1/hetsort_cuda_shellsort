#include "hetsort.cuh"

void sortThrust2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus) {
    #pragma omp parallel for num_threads(gpus.size())
    for (int g = 0; g < gpus.size(); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        gpu.initialize();

        for (size_t i = 0; i < chunks.size(); ++i) {
            int* mainBuffer = gpu.useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
            int* secondaryBuffer = gpu.useFirstBuffer ? gpu.buffer2 : gpu.buffer1;

            if (i == 0) 
                doubleMemcpy(mainBuffer, chunks[i].data(), chunks[i].size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);

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

        cudaStreamSynchronize(gpu.stream1);
        gpu.destroy();
    }
}

void sortThrust3N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus) {
    #pragma omp parallel for num_threads(gpus.size())
    for (int g = 0; g < gpus.size(); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        gpu.initialize();

        for (size_t i = 0; i < chunks.size(); ++i) {
            int* mainBuffer = gpu.useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
            int* secondaryBuffer = gpu.useFirstBuffer ? gpu.buffer2 : gpu.buffer1;
            cudaStream_t mainStream = gpu.useFirstBuffer ? gpu.stream1 : gpu.stream2;
            cudaStream_t secondaryStream = gpu.useFirstBuffer ? gpu.stream2 : gpu.stream1;

            if (i == 0) 
                doubleMemcpy(mainBuffer, chunks[i].data(), chunks[i].size(), cudaMemcpyHostToDevice, mainStream, gpu.streamTmp);

            thrustsort(mainBuffer, chunks[i].size(), gpu.bufferTmp, gpu.bufferSize, mainStream);

            if ((i > 0)) 
                doubleMemcpy(chunks[i - 1].data(), secondaryBuffer, chunks[i - 1].size(), cudaMemcpyDeviceToHost, secondaryStream, gpu.streamTmp);
            
            if (i + 1 < chunks.size()) 
                doubleMemcpy(secondaryBuffer, chunks[i + 1].data(), chunks[i + 1].size(), cudaMemcpyHostToDevice, secondaryStream, gpu.streamTmp);
            
            if (i == chunks.size() - 1) 
                cudaMemcpyAsync(chunks[i].data(), mainBuffer, chunks[i].size() * sizeof(int), cudaMemcpyDeviceToHost, mainStream);

            gpu.toggleBuffer();
        }
        
        cudaStreamSynchronize(gpu.stream1);
        gpu.destroy();
    }
}

void sortThrustInplace(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus) {
    #pragma omp parallel for num_threads(gpus.size())
    for (int g = 0; g < gpus.size(); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        gpu.initialize();

        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i == 0) 
                doubleMemcpy(gpu.buffer1, chunks[i].data(), chunks[i].size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);

            thrustsort(gpu.buffer1, chunks[i].size(), gpu.buffer2, gpu.bufferSize, gpu.stream1);

            doubleMemcpy(chunks[i].data(), gpu.buffer1, chunks[i].size(), cudaMemcpyDeviceToHost, gpu.stream1, gpu.streamTmp);

            if (i + 1 < chunks.size()) 
                doubleMemcpy(gpu.buffer1, chunks[i + 1].data(), chunks[i + 1].size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);
        }

        cudaStreamSynchronize(gpu.stream1);
        gpu.destroy();
    }
}

void sortThrustInplaceMemcpy(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus) {
    #pragma omp parallel for num_threads(gpus.size())
    for (int g = 0; g < gpus.size(); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        gpu.initialize();

        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i == 0)
                cudaMemcpyAsync(gpu.buffer1, chunks[i].data(), chunks[i].size() * sizeof(int), cudaMemcpyHostToDevice, gpu.stream1);

            thrustsort(gpu.buffer1, chunks[i].size(), gpu.buffer2, gpu.bufferSize, gpu.stream1);

            int* nextChunkData = (i + 1 < chunks.size()) ? chunks[i + 1].data() : nullptr;
            size_t nextChunkSize = (i + 1 < chunks.size()) ? chunks[i + 1].size() : 0;
            InplaceMemcpy(nextChunkData, gpu.buffer1, chunks[i].data(), nextChunkSize, chunks[i].size(), gpu.stream1, gpu.streamTmp);
        }

        cudaStreamSynchronize(gpu.stream1);
        gpu.destroy();
    }
}

void sortShell(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus) {
    #pragma omp parallel for num_threads(gpus.size())
    for (int g = 0; g < gpus.size(); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        gpu.initialize();

        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i == 0) {
                padVectorToPowerOfTwo(chunks[i]);
                doubleMemcpy(gpu.buffer1, chunks[i].data(), chunks[i].size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);
            }

            shellsort(gpu.buffer1, chunks[i].size(), gpu.stream1);

            doubleMemcpy(chunks[i].data(), gpu.buffer1, chunks[i].size(), cudaMemcpyDeviceToHost, gpu.stream1, gpu.streamTmp);

            if (i + 1 < chunks.size()) {
                padVectorToPowerOfTwo(chunks[i + 1]);
                doubleMemcpy(gpu.buffer1, chunks[i + 1].data(), chunks[i + 1].size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);
            }
        }

        cudaStreamSynchronize(gpu.stream1);
        gpu.destroy();
    }
}

void sortShell2N(std::vector<std::vector<std::vector<int>>>& chunkGroups, std::vector<GPUInfo>& gpus) {
    #pragma omp parallel for num_threads(gpus.size())
    for (int g = 0; g < gpus.size(); ++g) {
        auto& chunks = chunkGroups[g];
        GPUInfo& gpu = gpus[g];
        gpu.initialize();

        for (size_t i = 0; i < chunks.size(); ++i) {
            int* mainBuffer = gpu.useFirstBuffer ? gpu.buffer1 : gpu.buffer2;
            int* secondaryBuffer = gpu.useFirstBuffer ? gpu.buffer2 : gpu.buffer1;
            cudaStream_t mainStream = gpu.useFirstBuffer ? gpu.stream1 : gpu.stream2;
            cudaStream_t secondaryStream = gpu.useFirstBuffer ? gpu.stream2 : gpu.stream1;

            // Copy the first chunk data to the main GPU buffer
            if (i == 0) {
                padVectorToPowerOfTwo(chunks[i]);
                doubleMemcpy(mainBuffer, chunks[i].data(), chunks[i].size(), cudaMemcpyHostToDevice, mainStream, gpu.streamTmp);
            }

            // Sort the chunk on the main buffer
            shellsort(mainBuffer, chunks[i].size(), mainStream);

            // Copy the sorted chunk back and the next chunk to the secondary buffer
            if (i > 0) 
                doubleMemcpy(chunks[i - 1].data(), secondaryBuffer, chunks[i - 1].size(), cudaMemcpyDeviceToHost, secondaryStream, gpu.streamTmp);
            if (i + 1 < chunks.size()) {
                padVectorToPowerOfTwo(chunks[i + 1]);
                doubleMemcpy(secondaryBuffer, chunks[i + 1].data(), chunks[i + 1].size(), cudaMemcpyHostToDevice, secondaryStream, gpu.streamTmp);
            }

            // Copy the last sorted chunk back
            if (i == chunks.size() - 1) 
                doubleMemcpy(chunks[i].data(), mainBuffer, chunks[i].size(), cudaMemcpyDeviceToHost, mainStream, gpu.streamTmp);

            // Toggle the buffer for the next chunk
            gpu.toggleBuffer();
        }

        // Wait for all streams to finish
        cudaStreamSynchronize(gpu.stream1);
        cudaStreamSynchronize(gpu.stream2);

        // Destroy the GPU buffers and streams
        gpu.destroy();
    }
}

std::vector<int> sortKernel(const std::string& method, int* h_inputArray, size_t arraySize, std::vector<GPUInfo>& gpus) {
    std::vector<int> h_outputArray(arraySize);
    GPUInfo& gpu = gpus[0];
    gpu.initialize();

    if (method == "thrustsortKernel") {
        doubleMemcpy(gpu.buffer1, h_inputArray, arraySize, cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);
        thrustsort(gpu.buffer1, arraySize, gpu.buffer2, gpu.bufferSize, gpu.stream1);
    } else if (method == "shellsortKernel") {
        std::vector<int> vec_inputArray(h_inputArray, h_inputArray + arraySize);
        padVectorToPowerOfTwo(vec_inputArray);
        doubleMemcpy(gpu.buffer1, vec_inputArray.data(), vec_inputArray.size(), cudaMemcpyHostToDevice, gpu.stream1, gpu.streamTmp);
        shellsort(gpu.buffer1, vec_inputArray.size(), gpu.stream1);
    }

    doubleMemcpy(h_outputArray.data(), gpu.buffer1, arraySize, cudaMemcpyDeviceToHost, gpu.stream1, gpu.streamTmp);
    cudaStreamSynchronize(gpu.stream1);
    gpu.destroy();
    return h_outputArray;
}

