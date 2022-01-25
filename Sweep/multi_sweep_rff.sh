#!/bin/bash

#SBATCH --job-name=sweep_rff
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --mem=100M
#SBATCH --array=0-1

module load lang/python/anaconda/3.8.5-2021-AM

srun python multi_sweep_rff.py --param_idx=$SLURM_ARRAY_TASK_ID --verbose="True" --NO_TRIALS=1 --significance_threshold=0.1