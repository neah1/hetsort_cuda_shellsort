import re
import csv
import sys

# Open the text file
directory = sys.argv[1]
with open(f'{directory}/console_output.txt', 'r') as file:
    lines = file.readlines()

# Prepare to store the data
data = []
current_data = {}

# Regular expressions for extracting data
method_re = re.compile(r'Method: (\w+), Distribution: (\w+), Array Size: (\d+), Array Byte Size: (\d+) MB, Device Memory: (\d+) MB')
iteration_re = re.compile(r'Iteration (\d+): (\d+) ms')
gpu_count_re = re.compile(r'GPUs available: (\d+)')
chunk_count_re = re.compile(r'Number of chunks: (\d+)')

# Extract data
for line in lines:
    method_match = method_re.search(line)
    iteration_match = iteration_re.search(line)
    gpu_count_match = gpu_count_re.search(line)
    chunk_count_match = chunk_count_re.search(line)
    
    if method_match:
        # New method and distribution info found, reset current data
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
    elif iteration_match:
        # Map iterations to their respective times
        iteration_number = int(iteration_match.group(1)) + 1  # Convert to 1-based index
        current_data[f'Iteration {iteration_number}'] = iteration_match.group(2)
        
        # After the last iteration, store the complete set of data
        if iteration_number == 10:
            data.append(current_data.copy())

# Write data to CSV
csv_columns = ['Method', 'Distribution', 'Array size', 'Device memory', 'Array MB', 'Chunks', 'GPUs'] + [f'Iteration {i}' for i in range(1, 11)]

with open(f'{directory}/console_output.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(data)

print("Data has been written to console_output.csv")
