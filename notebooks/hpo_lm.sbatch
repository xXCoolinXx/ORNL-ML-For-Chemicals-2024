#!/bin/bash

#SBATCH -A birthright
#SBATCH -p burst
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -J hpo_lm
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH -o ./logs/slurmout/%j-test-output.txt
#SBATCH -e ./logs/slurmout/%j-test-error.txt
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fcf@ornl.gov

# export OMPI_MCA_opal_cuda_support=true
# module load PE-gnu/4.0

source ../../miniconda3/etc/profile.d/conda.sh
conda activate ../../miniconda3/envs/cool-collin

# nvcc --version
nvidia-smi

python hpo_lm.py