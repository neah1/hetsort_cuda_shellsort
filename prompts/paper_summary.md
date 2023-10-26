ABSTRACT
Sorting is a classic algorithmic problemand its importance has led to the design and implementation of various
sorting algorithms on many-core graphics processing units (GPUs). CUDPP Radix sort is the most efficient
sorting on GPUs and GPU Sample sort is the best comparison-based sorting. Although the implementations
of these algorithms are efficient, they either need an extra space for the data rearrangement or the atomic
operation for the acceleration. Sorting applications usually deal with a large amount of data, thus the memory
utilization is an important consideration. Furthermore, these sorting algorithms on GPUs without the atomic
operation support can result in the performance degradation or fail to work.
In this paper, an efficient implementation of a parallel shellsort algorithm, CUDA shellsort, is proposed for many-core GPUs with CUDA.
Experimental results show that, on average, the performance of CUDA shellsort is nearly twice faster than
GPU quicksort and 37% fasterthan Thrust mergesort under uniform distribution. Moreover, its performance
is the same as GPU sample sort up to 32 million data elements, but only needs a constant space usage. CUDA
shellsort is also robust over various data distributions and could be suitable for other many-core architectures

PRELIMINARY CONCEPTS FOR SHELLSORT
The shellsort algorithm can be written as Algorithm 1.
The shellsort is a generalization of the
insertion sort. It uses the insertion sort to sort K
interleaved subsequences of input data, where
K is an increment drawn from a predefined
integer sequence. The shellsort performs t passes over the
input sequence that each pass for an increment
from ht-1 to h0. A pass of shellsort
is defined as sorting all columns of the 2D array.
For each pass of shellsort, the number of
columns of the 2D array is diminished and the
number of rows of the 2D array is increased.
Until the final pass, the 2D array only has one
column, and the sorting phase on this column
is the same as the ordinary insertion sort. Thus,
the correctness of shellsort is also established.
The column width of 2D array is called an
increment. A sequence of increments forms an
increment sequence.
The performance of shellsort algorithm is
heavily depended on the choice of increment
sequence it used and there are many increment
sequences have been proposed in the literature
(Sedgewick, 1996). By our best knowledge, the
current best increment sequence is proposed by
Ciura (2001):
M_Ciura = {1, 4, 10, 23, 57, 132, 301, 701, 1750}.
Due to the hardness of complexity analysis
for the large increment, Marcin Ciura’s increment sequence terminates at 1750. However,
the next increment can be more than 2 times
than the previous one and a ratio of 2.2 times is adopted in the experimental tests

CUDA SHELLSORT
In this section, an overview of the CUDA
shellsort algorithm is given, and then the
implementation details are presented in the
following sections.
The Hybrid Sorting Architecture CUDA shellsort consists of three phases. For
an input sequence of size N:
• Phase 1-Parallel shellsort phase. For each
increment K, K threads concurrently sort K
subsequences of the input sequence. This
process continues until K is smaller than
a threshold S.
• Phase 2-Bitonic merge sort phase. The
input sequence is divided into [N/S]
subsequences. Each thread block loads its
own subsequence and sorts it by using the
bitonic merge sort.
• Phase 3-Odd-even bitonic merge phase.
The [N/S] subsequences are adjusted by
using the odd-even bitonic merge several
times.

The increment sequence used in the parallel
shellsort phase is obtained from Ciura (2001).
Each increment in the Marcin Ciura’s increment
sequence represents a shellsort pass and each
shellsort pass consists of a kernel program.
There are totally O(logN) increments for the
input sequence of size N, thus O(logN) kernel
programs are lunched in the parallel shellsort
phase. The sorting algorithm used in each thread
is the insertion sort.

For the ordinary parallel shellsort implementation (Breshears, 2009), the parallelism
of shellsort is lost iteration by iteration due to
the increment is decreased geometrically by
each iteration. Although the parallelism of
concurrent insertion sort to each disjoint subsequence is decreased; however, the workload
of each insertion sort is increased. This phenomenon is analogous to the parallel mergesort
that the data parallelism is surplus at the beginning of sorting phase; however, the parallelism
is decreased geometrically after each pairwisemerge. For current NVIDIA GPUs, there must
be at least 5000 concurrent threads to fully
occupy the underlying hardware’s computing
power (Satish, Harris, & Garland, 2009). The
parallelism lost problem results in low use of
computation resource, thus most of the SMs
are idle and the insufficient threads lead to
expose the global memory latency. To solve
this problem, we switch to the bitonic merge
sort when the parallelism in the parallel shellsort
phase reaches a threshold S.
In the bitnoic merge sort phase, the input
sequence is divided into [N/S] subsequences (blocks).
Then each thread block sorts its own block by using the bitonic merge
sort. Since the input size of the bitonic merge
sort must be some power of 2, we pad the final
block with maximum values when its length
does not equal to S. The final phase is to merge
[N/S] sorted blocks. We do not use the
mergesort-like algorithm to merge the sorted
blocks due to the parallelism lost problem inherently in the parallel mergesort. The sorted
block can be viewed as a single data element
and an odd-even transposition sort can be applied to sort these sorted blocks. The comparison operation between two sorted blocks in the
odd-even transposition sort is replaced with the
bitonic merge. In the odd pass, we merge the
odd-even pairs of sorted blocks; in the even
pass, we merge even-odd pairs of sorted blocks.
This is called the odd-even bitonic merge. The
problem comes up with the odd-even transposition sort is that it takes O(N) of odd-even iterations to sort a sequence with N data elements.
When N grows to very large, the odd-even
bitonic merge could be infeasible. Fortunately,
the large N is never occurred in this phase due
to the parallel shellsort phase has rearranged
the input sequence to a partially sorted order.
The partially sorted input sequence derived
from the parallel shellsort phase facilitates the
iterations of odd-even bitonic merge may be
bounded by a small constant. According to the
experimental tests, when a threshold S is set to
2048, at most eight iterations of odd-even bitonic merge are needed to merge [N / 2048] sorted blocks

Implementation Details
The CUDA programming model encourages a
two-level hierarchical view of threads (Nickolls,
Buck, Garland, & Skadron, 2008). The top level
consists of thread blocks; each thread block is
independent of others. However, within each
thread block, threads can communicate to each
other via the shared memory. The bottom level
consists of massively independent threads which
acts the same as the stream processing. CUDA
shellsort in the parallel shellsort phase focuses
on the bottom level view. In this point of view,
a sorting problem is divided into independent
subproblems, and each thread solves an instance of the subproblem by the same method.
In the bitonic merge sort and odd-even bitonic
merge phases, CUDA shellsort turns into the
thread-block level due to the communications
are needed within the threads of a thread block.

Parallel Shellsort Phase
For an input sequence Seq of size N, CUDA
shellsort first determines its starting increment,
K, from the Marcin Ciura’s increment sequence.
The kernel program generates K threads, and
then each thread sorts a subsequence of Seq by the insertion sort.
For example, the starting increment is 1750 with an input sequence
of size 3000, and then the kernel program
generates 1750 threads. The first thread sorts
a subsequence Seq1 = {Seq[0], Seq[1750]}, the second thread sorts a subsequence Seq2 = {Seq[1], Seq[1751]}, and so forth.

In order to ensure all subsequences are sorted before the next shellsort pass, a global synchronization is needed
by leaving and reentering the kernel program.
For this phase, we have some observations:
(1) the Marcin Ciura’s increment sequence has
a nice property that for each adjacent increment in the increment sequence, the number
of comparisons to insert a data item into a
sorted subsequence is no more than 20 by the
experimental tests. Therefore, we can maintain a
small amount of data in a fast memory used for
the fast comparison and exchange of inserting
operation. (2) In each shellsort pass, each thread
continues inserting an unsorted data item to a
sorted subsequence. The insertion time for a
sorted subsequence of length M can be improved
by maintaining a heap structure attached to each
thread, thus reduce the insertion time from O(M)
time complexity in the worst case to O(logM).
(3) Subsequences are disjoint and the sorting
processes are independent. Therefore, there is
no communication between threads.
It is a good chance to implement a heap
structure via the registers which are private to
each thread. Although the register cannot be
indexed and can be nonproductive from the
programmer’s perspective, for the performance
reason, a register file has some benefits than
the shared memory (Volkov & Demmel, 2008).
At the beginning of a shellsort pass, the heap
structure of each thread is filled up with first
21 data elements of a subsequence reside in
the global memory. The minimum data item is
then deleted from the heap structure and store
to the first block of input sequence, and then
the 22th data element is inserted into the heap
structure. All the global memory accesses are
coalesced and each data item accessed only
two times in a shellsort pass. The heap operation process repeats until all data elements in
a subsequence are processed.

Bitonic Merge Sort Phase
The increment used for each pass of shellsort
decreases geometrically. For the scalability, we
must generate as much threads as possible to
well utilize underlying hardware’s computing
power. In the CUDA shellsort implementation,
due to the size constraint of shared memory, an
increment 2048 is inserted to the Marcin Ciura’s
increment sequence, and then we switch to the
bitonic merge sort when finishing the shellsort
pass of the increment 2048 in the parallel shellsort phase. The reason to choose the increment
2048 is that this increment is a power of two
and still smaller than the size of shared memory. The reason to choose the bitonic merge
sort is due to it is an efficient comparison-based
sorting for the small data size (Blelloch, Leiserson, Maggs, Plaxton, Smith, & Zagha, 1998).
In the bitonic merge sort phase, for an input
sequence of size N, we divide the input sequence
to [N / 2048] blocks. We pad the maximum
data elements if the final block is smaller than
2048 elements. Then, the [N / 2048] thread
blocks are generated and each thread block sorts
its own 2048-element block. Besides, the
global memory accesses in this phase are coalesced and each data item is read/write once.

Odd-Even Bitonic Merge Phase
There are [N / 2048] sorted blocks though the
numerical range of adjacent sorted blocks may
be overlapped. Since the blocks are sorted, we
can adjust unordered data elements of two
concatenated sorted blocks into one sorted block
by using the bitonic merge. After the bitonic
merge on odd-even pairs of sorted blocks, the
even-odd pairs of sorted blocks may also be
overlapped with the common numerical range.
Therefore, the input sequence must be adjusted
with the bitonic merge on even-odd pairs of
sorted blocks again. According to the experimental tests, it needs at most eight times of
odd-even/even-odd bitonic merge to adjust
sorted 2048-element blocks from the output of
the bitonic merge sort in the bitonic merge sort
phase.
In details, we first generate [N / 2048]
thread blocks of 512 threads. Then, the two
adjacent 2048-element blocks are coalesced
read into eight registers of each thread of a
thread block. There are totally eight passes to
read two 2048-element blocks into eight registers of each thread of a thread block. For
example, in the first pass, each thread block
reads the first 512 data elements of first 2048-element block into the register R1 of each thread
within a thread block. In the second pass, each
thread block reads second 512 data elements
of first 2048-element block into the register R2
of each thread within a thread block. After
loading first 2048-element block, we proceed
with the second 2048-elements block. However, the second 2048-element block must be
read in reversely to form the bitonic sequence.
Thereafter, the register pairs of (R1, R5), (R2, R6), (R3,
R7), (R4, R8) of each thread are compared and
exchanged if out of order. This is the first pass
of bitonic merge of two 2048 bitonic sequences.
We call this inter thread-block bitonic merge
of (512, 4). Then, the inter block bitonic merge
is continuing with (512, 2) and (512, 1). After
the inter thread-block bitonic merge, we switch
to intra thread-block bitonic merge. The intra
thread block bitonic merge is the same as the
ordinary bitonic merge. It needs the communication within a thread block, hence the register
must firstly store to the shared memory before
the proceeding. The memory access patterns
in all phases are coalesced except for the first
phase is misaligned.
