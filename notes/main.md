### TODOs: Benchmarking

thrust comparison
cudamalloc in profile
test different bitonic CUDA sort

KERNEL SHELL VS THRUST:
+ performance comparison of kernel only + profile
+ Impact of Data Size and Distribution: show how dataset distribution affects performance

HET SHELL VS THRUST: 
+ performance comparison between different buffer strategies (end to end) + profile
+ Impact of Data Size and Distribution: show how dataset distribution affects performance
+ benchmark of separate phases of the algorithm for each strategy (memory transfer, GPU sorting, CPU merge phase, initialization overhead)
+ Inplace memory transfer VS Double stream memory transfer

PARADIS: 
+ performance comparison between the 3 main algorithms (only end to end)