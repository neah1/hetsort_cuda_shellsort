# parallel-shellsort

## Overview
Implementation of HETsort that leverages a parallel shellsort kernel optimized for NVIDIA GPUs, focusing on sorting high volumes of data in heterogeneous computing environments and optimizing device memory usage.

## Prerequisites
- NVIDIA GPU
- CUDA Toolkit 10.0+
- GCC: Version 8.5.0+
- GNU Make: 4.2.1+
- Python 3.6+

## Structure
- src/: Source files for the main application.
- profiles/: Directory for profile outputs.

## Compilation
``` bash
make build        # Compiles the source code into an executable named 'main'.
make run          # Builds (if necessary) and runs the application.
make clean        # Removes generated profile outputs.
make clean-sqlite # Removes SQLite files from profiles directory.
```

## Usage
``` bash
./main <method> <distribution> <arraySize> <deviceMemory> <iterations> <warmup> <checkSorted> <gpuCount> <seed>
```

### Parameters:
- method: Sorting method to use.
- distribution: Type of data distribution.
- arraySize: Number of elements to sort.
- deviceMemory: GPU memory to utilize.
- iterations: Number of times to run the benchmark.
- warmup: Number of warmup runs before timing.
- checkSorted: Whether to check if the array is sorted post-operation.
- gpuCount: Number of GPUs to use.
- seed: Random seed for data generation.

## Benchmarking

``` bash
./benchmark.sh
```
This script automates the running of different configurations to facilitate a comprehensive performance analysis.

``` bash
./generate_all.sh
```
This script utilizes the Python scripts generate_csv.py, generate_merge.py, and generate_nsys.py to process the results into csv format.

## Academic Context
This project is conducted as part of a Master's thesis in Computer Science at University of Oslo (UiO).
