#!/bin/bash
#SBATCH --job-name=gpu_python_job  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=25G                   # Memory allocation
#SBATCH --time=12:00:00             # Maximum execution time
#SBATCH --partition=mrigpu          # Use GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU


# Load necessary modules
module use /mnt/it_software/easybuild/modules/all
module load Python/3.12.3-GCCcore-13.3.0

# Navigate to your working directory
cd /mnt/mridata/jshen2/git/sparq
# python -m venv .venv
# Run Python script
nvidia-smi > nvidia.txt
