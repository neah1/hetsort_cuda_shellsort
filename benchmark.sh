#!/bin/bash

# Open output redirection
mkdir -p ./profiles
exec > >(tee ./profiles/console_output.txt) 2>&1

# Define arrays of parameters
methods=("thrustsort2N" "thrustsort3N" "thrustsortInplace" "shellsort" "shellsort2N")
memcpy_methods=("thrustsortInplaceMemcpy")
arraySizes=(100000000)
deviceMemories=(100)

kernel_methods=("shellsortKernel" "thrustsortKernel")
kernel_arraySizes=(100000000)

# Distributions: "uniform" "normal" "sorted" "reverse_sorted" "nearly_sorted"
distributions=("uniform") 

iterations=2
checkSorted=0
seed=42

# Function to run a benchmark with given parameters
run_benchmark() {
    local method=$1
    local distribution=$2
    local arraySize=$3
    local deviceMemory=$4
    local outputFile="./profiles/profile_${method}_${distribution}_${arraySize}_${deviceMemory}.nsys-rep"
    nsys profile --stats=true --output=$outputFile ./main $method $distribution $arraySize $deviceMemory $iterations $checkSorted $seed
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
            run_benchmark $method uniform $arraySize $deviceMemory
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
