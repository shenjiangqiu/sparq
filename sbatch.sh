#!/bin/bash
#SBATCH .-job-name=my_mri_job # Job name
#SBATCH .-nodes=1 # Number of nodes
#SBATCH .-partition=gpu # Partition name (use mrigpu for GPU jobs)
#SBATCH .-output=my_mri_job.out # Standard output and error log