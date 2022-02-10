#!/bin/bash

#SBATCH --job-name=sweep_rff
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=20:00:00
#SBATCH --mem=10G
#SBATCH --array=0-1

SCRIPT_DIR=/user/work/ll20823/mini-project

source $SCRIPT_DIR/projenv/bin/activate

module load lang/python/anaconda/3.8.5-2021-AM

python $SCRIPT_DIR/Sweep/multi_sweep_rff.py --param_idx=$SLURM_ARRAY_TASK_ID --verbose="True" --NO_TRIALS=1000 --significance_threshold=0.1 --ncpus=$SLURM_CPUS_PER_TASK

deactivate