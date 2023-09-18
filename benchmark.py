import time

from cuda_shellsort import cuda_shell_sort


def benchmark_cuda_shell_sort(data_sizes):
    execution_times = []

    for size in data_sizes:
        data = generate_test_data(size)  # Replace with your data generation function
        start_time = time.time()
        sorted_data = cuda_shell_sort(data)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append((size, execution_time))

    return execution_times
