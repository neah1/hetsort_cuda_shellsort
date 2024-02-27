## HET Sort overview (Thrust vs Shellsort)
1. Finds out number of GPUs and how much memory each GPU has. 
2. Takes an unsorted array, and splits it into chunks that fits into GPUs (unsorted array can be bigger than sum of GPU s memory, in which case each GPU will have to sort multiple chunks).
3. Copy chunks to GPU and run parallel shellsort
4. Once a GPU is done sorting, copy sorted chunk back to CPU
5. If two chunks have been copied back to to CPU, CPU starts merging the two chunks

## HET Sort optimization
- Splitting Strategy: It may be necessary to further split the dataset into chunks that can be sequentially processed by each GPU. This means each GPU might work on multiple chunks one after the other, effectively treating the GPU memory as a cache for parts of the dataset.
- Memory Management: The code allocates and frees GPU memory for each chunk. Depending on the size of your chunks and the available GPU memory, you might be able to optimize this by reusing allocated memory for multiple chunks.
- Asynchronous Execution: To fully leverage the capabilities of multiple GPUs, consider using CUDA streams for overlapping data transfers and computation, allowing different GPUs to work concurrently.
- Overlapping Computation and Transfer: When dealing with multiple chunks per GPU, you can overlap computation (sorting) on one chunk with the transfer of the next chunk to the GPU and the transfer of the previously sorted chunk back to the CPU. This requires careful management of CUDA streams and events to synchronize operations without stalling the GPU or CPU. 2N approach from Maltenberger and Stele & Jacobsen paper adapted for in-place sorting.

## HET Sort optimization (TODO)
- Thrust needs extra freemem (ensureCapacity).
- Deal with GPU variable free memory (splitting chunks logic). 
- Bi-directional copying inplaceMemCpy FROM MALTENBERGER.
- Reduce sync in inplacememcpy or implement ThreadPool for each GPU stream.
- Eager merging VS multi-way merging
- Merge Logic: Implementing an efficient merge logic that takes advantage of early chunk availability can be complex, especially if chunks finish at vastly different times. You may need a more sophisticated data structure to track sorted chunks and their merge partners.
- Pragma optimization for CPU