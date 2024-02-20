### 1. Parallel ShellSort Phase

-   For each increment \( K \), \( K \) threads concurrently sort \( K \) subsequences of the input sequence.
-   This process continues until \( K \) is smaller than a threshold \( S \).
-   The increment sequence used is obtained from Ciura (2001).

### 2. Bitonic Merge Sort Phase

-   The input sequence is divided into \([N/S]\) subsequences.
-   Each thread block loads its own subsequence and sorts it using the bitonic merge sort.

### 3. Odd-Even Bitonic Merge Phase

-   The \([N/S]\) subsequences are adjusted using the odd-even bitonic merge several times.


## Shellsort optimization
0. CUDA Shellsort optimization
Implement optimizations described in the paper such as bitonic and even odd sorting
1. Optimize Memory Access Patterns
Coalesced Access: Ensure that memory accesses within each warp are coalesced. Since each thread in a warp accesses consecutive memory locations when increment is 1, this is naturally coalesced. However, for larger increments, the access pattern might not be optimal. Aligning data accesses to achieve coalescence can significantly improve memory bandwidth utilization.
2. Minimize Divergence
Branch Divergence: The conditional check if (index >= increment) return; can lead to warp divergence, especially for larger increments where only a few threads in a block do meaningful work. Consider restructuring the kernel to minimize divergence, possibly by ensuring all threads in a warp can proceed with useful computation.
3. Dynamic Parallelism for Small Increments
Nested Parallelism: For small increments where the workload per thread becomes significantly lighter, consider using CUDA Dynamic Parallelism to launch nested kernels. This allows you to adjust the granularity of parallelism dynamically, potentially offloading final increments sorting to more optimized, possibly even different, sorting algorithms better suited for small data sets.
4. Adjust Thread and Block Sizes Dynamically
Adaptive Configuration: The static configuration of numThreads and numBlocks might not be optimal across all increment sizes. For larger increments, fewer threads might be necessary, while smaller increments could benefit from maximizing occupancy. Adjusting these dynamically based on increment size and profiling results can lead to better resource utilization.
5. Kernel Fusion for Small Gaps
Combine Operations: When the gap size becomes small, the overhead of launching a kernel becomes more significant relative to the computation performed by the kernel. If possible, consider fusing subsequent operations into a single kernel launch to reduce overhead. This is more of a complex optimization and may involve significant changes to the algorithm structure.
6. Efficient Data Transfer
Pinned Memory: If you're transferring data between host and device, using pinned (page-locked) memory can speed up these transfers. This is more relevant for the overall application structure than the specific kernel performance.
7. Utilize Shared Memory
Shared Memory for Temporary Storage: For segments of the array being sorted within a block, using shared memory can reduce global memory accesses, as shared memory is much faster. This requires careful management of synchronization within the block but can lead to substantial performance gains, especially for smaller increments where more data can be loaded into shared memory.
8. Profiling and Tuning
Detailed Profiling: Utilize tools like NVIDIA Nsight Compute and Nsight Systems for detailed profiling of your kernels. These tools can help identify bottlenecks, uncoalesced accesses, excessive divergence, and other issues. Based on profiling insights, further fine-tune block and thread sizes, memory usage, and algorithm phases.
9. Algorithmic Adjustments
Hybrid Approaches: For the final passes of the ShellSort, consider switching to a different sorting algorithm that may be more efficient for nearly sorted data or small arrays, such as insertion sort or bitonic sort, executed on the GPU.