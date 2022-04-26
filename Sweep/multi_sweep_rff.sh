#!/bin/bash

#SBATCH --job-name=sweep_rff
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=200:00:00
##SBATCH --mem=50G
#SBATCH --mem-per-cpu=15G
#SBATCH --array=0

SCRIPT_DIR=/user/work/ll20823/mini-project

source $SCRIPT_DIR/projenv/bin/activate

module load lang/python/anaconda/3.8.5-2021-AM

python $SCRIPT_DIR/rff/Sweep/multi_sweep_rff.py --param_idx=$SLURM_ARRAY_TASK_ID --verbose="True" --NO_TRIALS=1000 --significance_threshold=0.05 --ncpus=$SLURM_CPUS_PER_TASK --method=ciq

deactivate