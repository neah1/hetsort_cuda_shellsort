import pandas as pd

# Load the CSV files
directory = './profiles'
csv_file_1 = f'{directory}/console_output.csv'
csv_file_2 = f'{directory}/nsys_output.csv'

# Read the CSV files into DataFrames
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

# Find common columns
common_columns = df1.columns.intersection(df2.columns)

# Merge the DataFrames on the common columns
merged_df = pd.merge(df1, df2, on=common_columns.tolist(), how='outer')

# Optionally, save the merged DataFrame to a new CSV file
merged_df.to_csv(f'{directory}/merged_output.csv', index=False)
