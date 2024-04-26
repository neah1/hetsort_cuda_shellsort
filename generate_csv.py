import re
import csv
import sys

# Open the text file
directory = sys.argv[1]
with open(f'{directory}/console_output.txt', 'r') as file:
    lines = file.readlines()

# Regular expressions for extracting data
method_re = re.compile(r'Method: (\w+), Distribution: (\w+), Array Size: (\d+), Array Byte Size: (\d+) MB, Device Memory: (\d+) MB')
gpu_count_re = re.compile(r'GPUs available: (\d+)')
chunk_count_re = re.compile(r'Number of chunks: (\d+)')
array_split_re = re.compile(r'ArraySplit phase: (\d+) ms')
kernel_re = re.compile(r'Kernel phase: (\d+) ms')
merge_re = re.compile(r'Merge phase: (\d+) ms')
iteration_re = re.compile(r'Iteration (\d+): (\d+) ms')
profiling_re = re.compile(r'Profiling .*')

# Prepare to store the data
data = []
current_data = {}
current_iteration = {}

# Extract data
for line in lines[1:]:
    method_match = method_re.search(line)
    gpu_count_match = gpu_count_re.search(line)
    chunk_count_match = chunk_count_re.search(line)
    array_split_match = array_split_re.search(line)
    kernel_match = kernel_re.search(line)
    merge_match = merge_re.search(line)
    iteration_match = iteration_re.search(line)
    profiling_match = profiling_re.search(line)

    if method_match:
        current_data = {
            'Method': method_match.group(1),
            'Distribution': method_match.group(2),
            'Array size': method_match.group(3),
            'Array MB': method_match.group(4),
            'Device memory': method_match.group(5),
        }
    elif gpu_count_match:
        current_data['GPUs'] = gpu_count_match.group(1)
    elif chunk_count_match:
        current_data['Chunks'] = chunk_count_match.group(1)
    elif array_split_match:
        current_iteration['ArraySplit'] = array_split_match.group(1)
    elif kernel_match:
        current_iteration['Kernel'] = kernel_match.group(1)
    elif merge_match:
        current_iteration['Merge'] = merge_match.group(1)
    elif iteration_match:
        iteration_number = int(iteration_match.group(1)) + 1
        current_data[f'Iteration {iteration_number}'] = iteration_match.group(2)
        current_data.update({f'{key} {iteration_number}': value for key, value in current_iteration.items()})
        current_iteration = {}
    elif profiling_match:
        data.append(current_data.copy())
        current_data = {}

with open(f'{directory}/console_output.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print("Data has been written to console_output.csv")
