#!/bin/bash

#SBATCH --job-name=sweep_rff
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --mem=100M
#SBATCH --array=0-1

srun multi_sweep_rff --param_idx=$SLURM_ARRAY_TASK_ID --verbose="True" --NO_TRIALS=1 --significance_threshold=0.1