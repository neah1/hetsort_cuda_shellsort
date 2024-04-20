#!/bin/bash

# Define arrays of parameters
methods=("thrustsort2N" "thrustsort3N" "thrustsortInplace" "shellsort" "shellsort2N")
memcpy_methods=("thrustsortInplaceMemcpy")
kernel_methods=("shellsortKernel" "thrustsortKernel")
distributions=("uniform" "normal" "sorted" "reverse_sorted" "nearly_sorted")
arraySizes=(10000000)
kernel_arraySizes=(10000000)
deviceMemories=(50)

# Function to run a benchmark with given parameters
run_benchmark() {
    local method=$1
    local distribution=$2
    local arraySize=$3
    local deviceMemory=$4
    local iterations=1
    local warmup=0
    local seed=42
    local outputFile="./profiles/profile_${method}_${distribution}_${arraySize}_${deviceMemory}.nsys-rep"
    
    nsys profile --stats=true --output=$outputFile ./benchmark $method $distribution $arraySize $deviceMemory $iterations $warmup $seed
}

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
