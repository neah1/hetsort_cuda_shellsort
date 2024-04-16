### TODOs: Benchmarking
- Setup benchmarking framework for comparing algorithms, including graphs (with nsight etc).


Shellsort vs Thrustsort kernels:
- performance comparison of kernel only
- Impact of Data Size and Distribution: show how dataset distribution affects performance
- analyze memory consumption patterns of kernels (space usage)

HETSort with shellsort kernel: 
- performance comparison between different buffer strategies (end to end)
- benchmark of separate phases of the algorithm for each strategy (memory transfer, GPU sorting, CPU merge phase, initialization overhead)
- Analyze memory consumption patterns at hybrid level (buffer/chunk size)

HETSort with thrustsort kernel:
- performance comparison between different buffer strategies (end to end)
- benchmark of separate phases of the algorithm for each strategy (memory transfer, GPU sorting, CPU merge phase, initialization overhead)
- Analyze memory consumption patterns at hybrid level (buffer/chunk size)

HETSort shellsort vs thrustsort: 
- performance comparison between best shellsort kernel and best thrustsort kernel
- Impact of Data Size and Distribution: show how dataset distribution affects performance
- analyze how much of shellsort extra sorting time is hidden by hybrid

PARADIS:
- Impact of Data Size and Distribution: show how dataset distribution affects performance

HETSort shellsort vs thrustsort vs PARADIS: 
- performance comparison between the 3 main algorithms
- multiway merge and PARADIS on CPU are not inplace (need extra mem).

Performance comparison of Inplace memory transfer VS Double stream memory transfer