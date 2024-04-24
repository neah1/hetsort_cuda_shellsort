#!/bin/bash

# Define arrays of parameters
methods=("thrustsort2N" "thrustsort3N" "thrustsortInplace" "shellsort" "shellsort2N")
arraySizes=(500000000 1000000000 2000000000 3000000000 4000000000 5000000000)
deviceMemories=(1000 2000 4000)

kernel_methods=("shellsortKernel" "thrustsortKernel")
kernel_arraySizes=(300000000 400000000 500000000 1000000000)

distributions=("uniform" "normal" "sorted" "reverse_sorted" "nearly_sorted") 

iterations=10
warmup=1
checkSorted=0
seed=42

# Function to run a benchmark with given parameters
run_benchmark() {
    local method=$1
    local distribution=$2
    local arraySize=$3
    local deviceMemory=$4
    local outputFile="./profiles/profile_${method}_${distribution}_${arraySize}_${deviceMemory}.nsys-rep"
    if [ -f "$outputFile" ]; then
        echo "Skipping profiling for $outputFile as it already exists."
    else
        nsys profile --stats=true --output=$outputFile ./main $method $distribution $arraySize $deviceMemory $iterations $warmup $checkSorted $seed 2>&1 | tee -a ./profiles/console_output.txt
    fi
}

mkdir -p ./profiles

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
for arraySize in "${arraySizes[@]}"; do
    for deviceMemory in "${deviceMemories[@]}"; do
        run_benchmark thrustsortInplaceMemcpy uniform $arraySize $deviceMemory
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
