## HET Sort overview (Thrust vs Shellsort)
1. Finds out number of GPUs and how much memory each GPU has. 
2. Takes an unsorted array, and splits it into chunks that fits into GPUs (unsorted array can be bigger than sum of GPU s memory, in which case each GPU will have to sort multiple chunks).
3. Copy chunks to GPU and run parallel shellsort
4. Once a GPU is done sorting, copy sorted chunk back to CPU
5. If two chunks have been copied back to to CPU, CPU starts merging the two chunks

## HET Sort optimization
- Error Checking: In a complete application, you should check the return values of CUDA API calls for errors.
- Memory Management: The code allocates and frees GPU memory for each chunk. Depending on the size of your chunks and the available GPU memory, you might be able to optimize this by reusing allocated memory for multiple chunks.
- Asynchronous Execution: To fully leverage the capabilities of multiple GPUs, consider using CUDA streams for overlapping data transfers and computation, allowing different GPUs to work concurrently.

## RETHINK MERGING CPU STRATEGY:
- Eager merging, tree merging etc. 
- Do we merge two chunks into a bigger chunk, and only merge chunks of the same size? or just randomly merge any two available chunks.
- Merge Logic: Implementing an efficient merge logic that takes advantage of early chunk availability can be complex, especially if chunks finish at vastly different times. You may need a more sophisticated data structure to track sorted chunks and their merge partners.
- Synchronization Overhead: Using cudaEventSynchronize() introduces synchronization points that can stall the CPU. Depending on how your application is structured, it might be beneficial to perform these checks in a non-blocking manner or on a separate thread to avoid stalling other CPU-side computations.