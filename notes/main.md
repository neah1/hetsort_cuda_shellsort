### TODOs: Benchmarking
- Setup benchmarking framework for comparing algorithms, including graphs.
- Create methods to generate datasets with different distribution according to essay for testing.
- In hetsort, data transfer time vs gpu sort time. Also include total end-to-end
- Use GB size instead of arraysize as main algorithm parameter?
- Compare with paradis

## Profiling and Tuning
- Utilize NVIDIA Nsight Compute and Nsight Systems for detailed profiling of your kernels. These tools can help identify bottlenecks, uncoalesced accesses, excessive divergence, and other issues. Based on profiling insights, further fine-tune block and thread sizes, memory usage, and algorithm phases.

- limitations: lack of c++ knowledge. Rewriting due to bad practices discovered during optimization.