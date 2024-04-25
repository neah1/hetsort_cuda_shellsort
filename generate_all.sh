#!/bin/bash

directory="./profiles"
output_file="$directory/nsys_output.txt"

# Initialize the output file with headers (assuming the headers from the first file are okay to use)
first_file=$(ls $directory/*.nsys-rep | head -n 1)
nsys stats --report nvtxsum,cudaapisum,gpukernsum,gpumemtimesum --format csv $first_file > $output_file

# Process each .nsys-rep file
for file in $directory/*.nsys-rep; do
    if [ "$file" != "$first_file" ]; then
        nsys stats --report nvtxsum,cudaapisum,gpukernsum,gpumemtimesum --format csv $file | tail -n +2 >> $output_file
    fi
done

echo "Data aggregated into nsys_output.txt"

python3 generate_csv.py
python3 generate_nsys.py
python3 generate_merge.py