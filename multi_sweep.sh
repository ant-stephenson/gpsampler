#!/bin/bash

#SBATCH --job-name=sweep_rff_precon
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH --array=1

SCRIPT_DIR=/user/work/ll20823/mini-project

source $SCRIPT_DIR/projenv/bin/activate

module load lang/python/anaconda/3.8.5-2021-AM

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.8.5-2021-AM/lib

python $SCRIPT_DIR/rff/multi_sweep.py --job_id $SLURM_ARRAY_JOB_ID \
    --param_idx=$SLURM_ARRAY_TASK_ID --verbose="True" --NO_TRIALS=1000 \
    --significance_threshold=0.1 --ncpus=$SLURM_CPUS_PER_TASK \
    --method=chol --pre="True"

deactivate