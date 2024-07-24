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
#SBATCH --array=0-35
#SBATCH --mail-user=$USER@bristol.ac.uk
#SBATCH --mail-type=END
#SBATCH --output=/user/work/ll20823/mini-project/slurm/sampling/test-%j_%a.out

source slurm_init.sh

echo $SLURM_JOB_START_TIME

DIMS=(2 5 10 20 50)
LENSCALES=(0.5 1.0 3.0 5.0)
KERNS=("rbf" "exp" "matern32" "matern52")

$(python  get_bash_dlk.py -d ${DIMS[@]} -ls ${LENSCALES[@]} -kt ${KERNS[@]} -arr $SLURM_ARRAY_TASK_ID)

echo $DIM $LENSCALE $KERN

if [ $KERN == "matern32" ]; then
  KERN="matern"
  NU=1.5
elif [ $KERN == "matern52" ]; then
  KERN="matern"
  NU=2.5
elif [ $KERN == "matern72" ]; then
  KERN="matern"
  NU=3.5
else
  NU=-999
fi

KTRUE=$KERN
KMODEL=$KERN
ID=9

if [ $KERN == "matern" ]; then
    INFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KTRUE}_dim_${DIM}_ls_${LENSCALE}_nu_${NU}_${ID}.npy" 
else
    INFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KTRUE}_dim_${DIM}_ls_${LENSCALE}_${ID}.npy" 
fi
OUTFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/data_test_outputs.csv" 

python $SCRIPT_DIR/gpsampler/run_hypothesis_test.py --m 10000 --dimension $DIM \
    --lengthscale $LENSCALE --kernel_type $KMODEL --method "rff" \
    --out $OUTFILE --id $ID --significance 0.1 --filepath $INFILE \
    --nu $NU


deactivate
