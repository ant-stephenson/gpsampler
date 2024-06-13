#!/bin/bash

#SBATCH --job-name=test_data
#SBATCH --partition=compute
#SBATCH --account=math026082 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --array=0
#SBATCH --mail-user=$USER@bristol.ac.uk
#SBATCH --mail-type=END
#SBATCH --output=/user/work/ll20823/mini-project/slurm/sampling/test-%j_%a.out


SCRIPT_DIR=/user/work/ll20823/mini-project

source slurm_init.sh

echo $SLURM_JOB_START_TIME

OUTFILE="${SCRIPT_DIR}/synthetic-datasets/CIQ_GENERATION_RESULTS/data_test_outputs_all.csv" 

for INFILE in $SCRIPT_DIR/synthetic-datasets/CIQ_GENERATION_RESULTS/*.npy; do

    python $SCRIPT_DIR/gpsampler/run_hypothesis_test.py --m 10000 \
        --method "rff" \
        --out $OUTFILE --significance 0.1 --filepath $INFILE

done
deactivate
