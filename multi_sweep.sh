#!/bin/bash

#SBATCH --job-name=bv_sweep_rff
#SBATCH --output=/user/work/ll20823/mini-project/slurm/rff/%x.%j.out
#SBATCH --error=/user/work/ll20823/mini-project/slurm/rff/%x.%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH --array=2
#SBATCH --account=math038284



SCRIPT_DIR=/user/work/ll20823/mini-project/gpsampler

module load languages/python/3.8.20

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.8.5-2021-AM/lib

source $SCRIPT_DIR/.ve38/bin/activate

python $SCRIPT_DIR/multi_sweep.py --job_id $SLURM_ARRAY_JOB_ID \
    --param_idx=$SLURM_ARRAY_TASK_ID --verbose="True" --NO_TRIALS=1000 \
    --significance_threshold=0.1 --ncpus=$SLURM_CPUS_PER_TASK \
    --method=rff --pre="True" --bv

deactivate