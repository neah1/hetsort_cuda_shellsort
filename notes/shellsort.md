## COMPLETED
6. Efficient Data Transfer
Pinned Memory: If you're transferring data between host and device, using pinned (page-locked) memory can speed up these transfers. This is more relevant for the overall application structure than the specific kernel performance.
7. Utilize Shared Memory (BITONIC SORT)
Shared Memory for Temporary Storage: For segments of the array being sorted within a block, using shared memory can reduce global memory accesses, as shared memory is much faster. This requires careful management of synchronization within the block but can lead to substantial performance gains, especially for smaller increments where more data can be loaded into shared memory.
4. Adjust Thread and Block Sizes Dynamically
Adaptive Configuration: The static configuration of numThreads and numBlocks might not be optimal across all increment sizes. For larger increments, fewer threads might be necessary, while smaller increments could benefit from maximizing occupancy. Adjusting these dynamically based on increment size and profiling results can lead to better resource utilization.
2. Minimize Divergence
Branch Divergence: The conditional check if (index >= increment) return; can lead to warp divergence, especially for larger increments where only a few threads in a block do meaningful work. Consider restructuring the kernel to minimize divergence, possibly by ensuring all threads in a warp can proceed with useful computation.
1. Optimize Memory Access Patterns
Coalesced Access: Ensure that memory accesses within each warp are coalesced. Since each thread in a warp accesses consecutive memory locations when increment is 1, this is naturally coalesced. However, for larger increments, the access pattern might not be optimal. Aligning data accesses to achieve coalescence can significantly improve memory bandwidth utilization. Solved cuz in shellsort all threads run independtly. Jumping across increments makes warp coalescence impossible.

### 1. Parallel ShellSort Phase
- For each increment \( K \), \( K \) threads concurrently sort \( K \) subsequences of the input sequence.
- This process continues until \( K \) is smaller than a threshold \( S \).
- The increment sequence used is obtained from Ciura (2001).

### 2. Bitonic Merge Sort Phase
- The input sequence is divided into \([N/S]\) subsequences.
- Each thread block loads its own subsequence and sorts it using the bitonic merge sort.

### 3. Odd-Even Bitonic Merge Phase
- The \([N/S]\) subsequences are adjusted using the odd-even bitonic merge several times.
- Cant write to registers in CUDA. Need to load in shared memory. Cant implement it in an efficient way.
