#!/bin/bash

methods=("thrustsort2N" "thrustsort3N" "thrustsortInplace" "thrustsortInplaceMemcpy" "shellsort" "shellsort2N")

arraySizes=(1000000000 2000000000 4000000000 6000000000 8000000000 10000000000)
deviceMemories=(2000 4000)

arraySizes2=(10000 100000 1000000 10000000 100000000)
deviceMemories2=(100 500 1000)

distributions=("uniform" "normal" "sorted" "reverse_sorted" "nearly_sorted") 
kernel_methods=("shellsortKernel" "thrustsortKernel")
kernel_arraySizes=(1000000000)

iterations=5
warmup=1
checkSorted=0
gpuCount=4
seed=42

mkdir -p ./profiles
run_benchmark() {
    local method=$1
    local distribution=$2
    local arraySize=$3
    local deviceMemory=$4
    local outputFile="./profiles/profile_${method}_${distribution}_${arraySize}_${deviceMemory}.nsys-rep"
    if [ -f "$outputFile" ]; then
        if nsys stats --report gpukernsum $outputFile | grep -q "SKIPPED"; then
            echo "No CUDA kernel data found in $outputFile"
            run_profile
        fi
    else
        run_profile
    fi
}

run_profile() {
    echo "Profiling $method $distribution $arraySize $deviceMemory"
    nsys profile --stats=true --force-overwrite=true --output=$outputFile ./main $method $distribution $arraySize $deviceMemory $iterations $warmup $checkSorted $gpuCount $seed 2>&1 | tee -a ./profiles/console_output.txt
}

# Benchmark kernels only
for distribution in "${distributions[@]}"; do
    for arraySize in "${kernel_arraySizes[@]}"; do
        for method in "${kernel_methods[@]}"; do
            run_benchmark $method $distribution $arraySize 0
        done
    done
done

# Standard benchmarks loop
for deviceMemory in "${deviceMemories[@]}"; do
    for arraySize in "${arraySizes[@]}"; do
        for method in "${methods[@]}"; do
            run_benchmark $method uniform $arraySize $deviceMemory
        done
    done
done

# Standard benchmarks loop
for deviceMemory in "${deviceMemories2[@]}"; do
    for arraySize in "${arraySizes2[@]}"; do
        for method in "${methods[@]}"; do
            run_benchmark $method uniform $arraySize $deviceMemory
        done
    done
done