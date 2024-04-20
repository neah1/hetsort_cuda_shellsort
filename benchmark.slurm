#!/bin/bash
#SBATCH --job-name=GPU_Benchmarks
#SBATCH --output=./outputs/benchmark_%j.out
#SBATCH --error=./outputs/benchmark_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# Clear environment and set safety flags
module purge
set -o errexit
set -o nounset
set -o pipefail

# Load necessary modules
module load cuda/10.2
module load gcc/10
module load nvidia/nsight-systems

# Execute the script
./benchmark.sh
