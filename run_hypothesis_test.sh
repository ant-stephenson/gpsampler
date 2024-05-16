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
#SBATCH --array=0-8
#SBATCH --mail-user=$USER@bristol.ac.uk
#SBATCH --mail-type=END
#SBATCH --output=/user/work/ll20823/mini-project/slurm/sampling/test-%j_%a.out


SCRIPT_DIR=/user/work/ll20823/mini-project

module load lang/python/anaconda/3.8.5-2021-AM

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.8.5-2021-AM/lib

source $SCRIPT_DIR/projenv/bin/activate

echo $SLURM_JOB_START_TIME

DIMS=(10 50 100)
DIMS+=( "${DIMS[@]}" "${DIMS[@]}" )
LENSCALES=(0.5 0.5 0.5 1.0 1.0 1.0 3.0 3.0 3.0)

DIM="${DIMS[$SLURM_ARRAY_TASK_ID]}"
LENSCALE="${LENSCALES[$SLURM_ARRAY_TASK_ID]}"
KTRUE="laplacian"
KMODEL="exp"
ID=99
LENSCALE=0.9

INFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KTRUE}_dim_${DIM}_ls_${LENSCALE}_${ID}.npy" 
OUTFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/data_test_outputs.csv" 

python $SCRIPT_DIR/gpsampler/run_hypothesis_test.py --m 10000 --dimension $DIM \
    --lengthscale $LENSCALE --kernel_type $KMODEL --method "rff" \
    --out $OUTFILE --id $ID --significance 0.1 --filepath $INFILE


deactivate
