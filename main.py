from benchmark import benchmark_cuda_shell_sort

if __name__ == "__main__":
    data_sizes = [1000, 5000, 10000]  # Add more sizes as needed
    execution_times = benchmark_cuda_shell_sort(data_sizes)

    for size, time in execution_times:
        print(f"Data Size: {size}, Execution Time: {time} seconds")
