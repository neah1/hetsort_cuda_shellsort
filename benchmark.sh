#!/bin/bash

# Redirect output to both file and console.
exec > >(tee console_output.txt) 2>&1

# Function to run a benchmark with given parameters
run_benchmark() {
    local method=$1
    local distribution=$2
    local arraySize=$3
    local deviceMemory=$4
    local outputFile="./profiles/profile_${method}_${distribution}_${arraySize}_${deviceMemory}.nsys-rep"
    nsys profile --stats=true --output=$outputFile ./benchmark $method $distribution $arraySize $deviceMemory $iterations $warmup $seed
}

# Define parameters
arraySizes=(50000000)
kernel_arraySizes=(50000000)
deviceMemories=(50)
iterations=2
warmup=1
seed=42

# Define arrays of parameters
methods=("thrustsort2N" "thrustsort3N" "thrustsortInplace" "shellsort" "shellsort2N")
kernel_methods=("shellsortKernel" "thrustsortKernel")
memcpy_methods=("thrustsortInplaceMemcpy")
distributions=("uniform" "normal" "sorted" "reverse_sorted" "nearly_sorted")

# Standard benchmarks loop
for method in "${methods[@]}"; do
    for distribution in "${distributions[@]}"; do
        for arraySize in "${arraySizes[@]}"; do
            for deviceMemory in "${deviceMemories[@]}"; do
                run_benchmark $method $distribution $arraySize $deviceMemory
            done
        done
    done
done

# Benchmark memcpy methods
for method in "${memcpy_methods[@]}"; do
    for arraySize in "${arraySizes[@]}"; do
        for deviceMemory in "${deviceMemories[@]}"; do
            run_benchmark $method "uniform" $arraySize $deviceMemory
        done
    done
done

# Benchmark kernels only
for method in "${kernel_methods[@]}"; do
    for distribution in "${distributions[@]}"; do
        for arraySize in "${kernel_arraySizes[@]}"; do
            run_benchmark $method $distribution $arraySize 0
        done
    done
done
