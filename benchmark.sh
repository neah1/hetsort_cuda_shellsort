#!/bin/bash

# Open output redirection
exec > console_output.txt 2>&1

# Function to run a benchmark with given parameters
run_benchmark() {
    local method=$1
    local distribution=$2
    local arraySize=$3
    local deviceMemory=$4
    local outputFile="./build/profile_${method}_${distribution}_${arraySize}_${deviceMemory}.nsys-rep"

    echo "Profiling method: $method, Distribution: $distribution, Array Size: $arraySize, Device Memory: $deviceMemory"
    nsys profile --stats=true --output=$outputFile ./build/main $method $distribution $arraySize $deviceMemory
}

# Define arrays of parameters
methods=("thrustsort2N" "thrustsort3N" "thrustsortInplace" "shellsort" "shellsort2N")
memcpy_methods=("thrustsortInplaceMemcpy")
kernel_methods=("shellsortKernel" "thrustsortKernel")
distributions=("uniform" "normal" "sorted" "reverse_sorted" "nearly_sorted")
arraySizes=(1000000)
kernel_arraySizes=(1000000)
deviceMemories=(100 200)

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
