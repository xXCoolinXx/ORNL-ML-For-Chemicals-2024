#!/bin/bash

#SBATCH -A birthright
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH  --gpus-per-node 4
#SBATCH -n 4
#SBATCH -c 8
#SBATCH -J gnn_hpo
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o ./logs/slurmout/%j-test-output.txt
#SBATCH -e ./logs/slurmout/%j-test-error.txt
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fcf@ornl.gov

set -x 

export OMPI_MCA_opal_cuda_support=true
export N_JOBS=2
module load PE-gnu/4.0

source ../../miniconda3/etc/profile.d/conda.sh
conda activate ../../miniconda3/envs/cool-collin

# nvcc --version
nvidia-smi

# mpirun -np $SLURM_NTASKS python lm_gnn_optuna.py

BASE_PORT=8889

for i in $(seq 1 $SLURM_NTASKS)
do
    # Calculate the port number for this run
    export GPU_ID=$i
    PORT=$((BASE_PORT + i))
    
    # Set the environment variable for the port
    export HYDRAGNN_MASTER_PORT=$PORT
    
    echo "Running mpirun with port $PORT"
    
    export CUDA_VISIBLE_DEVICES=$((i - 1))
    # Run your MPI program
    mpirun -np 1 python -u gnn_optuna.py &
    # | tee out.log
    # Optional: add a small delay between runs
    # sleep 1
done

wait
# for i in {1, $SLURM_NTASKS}; do
    # mpirun python lm_gnn_optuna.py &
# done
# wait
# srun --mpi=openmpi python lm_gnn_optuna.py