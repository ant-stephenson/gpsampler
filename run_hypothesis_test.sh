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

source slurm_init.sh

echo $SLURM_JOB_START_TIME

DIMS=(2 5 10)
LENSCALES=(0.5 2.0 5.0)
KERNS=("matern")

$(python  get_bash_dlk.py -d ${DIMS[@]} -ls ${LENSCALES[@]} -kt ${KERNS[@]} -arr $SLURM_ARRAY_TASK_ID)

KTRUE=$KERN
KMODEL=$KERN
ID=3

if [ $KERN == "matern" ]; then
    NU=2.5
    INFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KTRUE}_dim_${DIM}_ls_${LENSCALE}_nu_${NU}_${ID}.npy" 
else
    NU=-999
    INFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KTRUE}_dim_${DIM}_ls_${LENSCALE}_${ID}.npy" 
fi
OUTFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/data_test_outputs.csv" 

python $SCRIPT_DIR/gpsampler/run_hypothesis_test.py --m 10000 --dimension $DIM \
    --lengthscale $LENSCALE --kernel_type $KMODEL --method "rff" \
    --out $OUTFILE --id $ID --significance 0.1 --filepath $INFILE \
    --nu $NU


deactivate
