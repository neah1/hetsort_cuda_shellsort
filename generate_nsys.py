import re
import csv
import sys

# Open the text file
directory = sys.argv[1]
with open(f'{directory}/nsys_output.txt', 'r') as file:
    data = file.read()

# Function to parse each block of data
def parse_data_block(block):
    # Extract the header line
    header_regex = (r"\[.*reports/(.*).py ./.*/profile_(.*?)_(uniform|normal|sorted|reverse_sorted|nearly_sorted)_(.*?)_(.*?).sqlite\]")
    header_match = re.search(header_regex, block)
    if not header_match:
        return []

    report_type, method, distribution, array_size, device_memory = header_match.groups()

    # Extract data lines
    lines = block.split('\n')[3:-1]  # Skip the first two lines and the last empty line
    results = []
    for line in lines:
        if line.strip():  # Ensure the line is not empty
            fields = line.split(',')

            if report_type == "gpukernsum":
                name_field = ','.join(fields[8:]).strip()
            else:
                name_field = fields[-1].strip()

            standardized_row = [
                method, distribution, array_size, device_memory, report_type, *fields[:8], name_field
            ]
            results.append(standardized_row)
    return results

# Parse all blocks separated by "Running"
blocks = data.split("Running ")[1:]  # Skip the first split which will be empty

# Aggregate all results
all_results = []
for block in blocks:
    all_results.extend(parse_data_block(block))

# Write to CSV
with open(f'{directory}/nsys_output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(['Method', 'Distribution', 'Array size', 'Device memory', 'Report type', 
                     'Time (%)', 'Total Time (ns)', 'Instances', 'Avg (ns)', 
                     'Med (ns)', 'Min (ns)', 'Max (ns)', 'StdDev (ns)', 'Name'])
    # Write the data
    for result in all_results:
        writer.writerow(result)

print("Data has been written to nsys_output.csv")
