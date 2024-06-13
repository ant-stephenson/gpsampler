#!/bin/bash

#SBATCH --job-name=sample_rff
#SBATCH --partition=compute
#SBATCH --account=math026082 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --array=0-2
#SBATCH --mail-user=$USER@bristol.ac.uk
#SBATCH --mail-type=END
#SBATCH --output=/user/work/ll20823/mini-project/slurm/sampling/rff-%j_%a.out

source slurm_init.sh

DIMS=(10)
LENSCALES=(0.5 2.0 5.0)
KERNS=("rbf")

$(python  get_bash_dlk.py -d ${DIMS[@]} -ls ${LENSCALES[@]} -kt ${KERNS[@]} -arr $SLURM_ARRAY_TASK_ID)

NU=2.5

N=110000
KS=0.9
NV=0.1

ID=3

# 1: 3 * n/nv
# 2: 12 * n/nv
# 3: n**3/2 / nv

OUTFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KERN}_dim_${DIM}_ls_${LENSCALE}_${ID}.npy" 

if [ $KERN == "exp" ]; then
  python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n $N --dimension $DIM \
    --lengthscale $LENSCALE --outputscale $KS --noise_variance $NV \
    --kernel_type "matern" --nu 0.5 \
    --out $OUTFILE --id $ID
elif [ $KERN == "rbf" ]; then
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n $N --dimension $DIM \
        --lengthscale $LENSCALE --outputscale $KS --noise_variance $NV --kernel_type "rbf" \
        --out $OUTFILE --id $ID
elif [ $KERN == "matern" ]; then
    OUTFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KERN}_dim_${DIM}_ls_${LENSCALE}_nu_${NU}_${ID}.npy" 
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n $N --dimension $DIM \
        --lengthscale $LENSCALE --outputscale $KS --noise_variance $NV \
        --kernel_type "matern" --nu $NU \
        --out $OUTFILE --id $ID
else
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n $N --dimension $DIM \
        --lengthscale $LENSCALE --outputscale $KS --noise_variance $NV \
        --kernel_type $KERN \
        --out $OUTFILE --id $ID
fi

conda deactivate
