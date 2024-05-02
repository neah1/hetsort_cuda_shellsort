#!/bin/bash

# "shellsort" "shellsort2N" "thrustsort2N" "thrustsort3N" "thrustsortInplace" "thrustsortInplaceMemcpy" "shellsortKernel" "thrustsortKernel"
# "uniform" "normal" "sorted" "reverse_sorted" "nearly_sorted"

# LARGE PROFILE RUN
# methods=("all")
# distributions=("uniform") 
# arraySizes=(10000000000)
# deviceMemories=(4096)
# iterations=1
# warmup=0
# checkSorted=0
# gpuCount=4
# seed=42
# profile=1

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
    echo "Profiling $method $distribution $arraySize $deviceMemory" | tee -a ./profiles/console_output.txt
    if [ $profile -eq 1 ]; then
        nsys profile --force-overwrite=true --output=$outputFile ./main $method $distribution $arraySize $deviceMemory $iterations $warmup $checkSorted $gpuCount $seed 2>&1 | tee -a ./profiles/console_output.txt
    else
        ./main $method $distribution $arraySize $deviceMemory $iterations $warmup $checkSorted $gpuCount $seed 2>&1 | tee -a ./profiles/console_output.txt
    fi   
}


for deviceMemory in "${deviceMemories[@]}"; do
    for arraySize in "${arraySizes[@]}"; do
        for distribution in "${distributions[@]}"; do
            for method in "${methods[@]}"; do
                run_benchmark $method $distribution $arraySize $deviceMemory
            done
        done
    done
done


