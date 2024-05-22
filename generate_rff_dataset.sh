#!/bin/bash

#SBATCH --job-name=sample_rff
#SBATCH --partition=compute
#SBATCH --account=math026082 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --array=0-11
#SBATCH --mail-user=$USER@bristol.ac.uk
#SBATCH --mail-type=END
#SBATCH --output=/user/work/ll20823/mini-project/slurm/sampling/rff-%j_%a.out

source slurm_init.sh

DIMS=(2 10 20 50)
DIMS+=( "${DIMS[@]}" "${DIMS[@]}" )
LENSCALES=(0.5 0.5 0.5 1.0 1.0 1.0 3.0 3.0 3.0)
LENSCALES=(3.0 3.0 3.0 5.0 5.0 5.0 7.0 7.0 7.0)

DIM="${DIMS[$SLURM_ARRAY_TASK_ID]}"
LENSCALE="${LENSCALES[$SLURM_ARRAY_TASK_ID]}"
KTYPE="matern"
NU=1.5

N=110000
KS=0.9
NV=0.1

ID=0

# 1: 3 * n/nv
# 2: 12 * n/nv
# 3: n**3/2 / nv

OUTFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KTYPE}_dim_${DIM}_ls_${LENSCALE}_${ID}.npy" 

if [ $KTYPE == "exp" ]; then
  python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n $N --dimension $DIM \
    --lengthscale $LENSCALE --outputscale $KS --noise_variance $NV \
    --kernel_type "matern" --nu 0.5 \
    --out $OUTFILE --id $ID
elif [ $KTYPE == "rbf" ]; then
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n $N --dimension $DIM \
        --lengthscale $LENSCALE --outputscale $KS --noise_variance $NV --kernel_type "rbf" \
        --out $OUTFILE --id $ID
elif [ $KTYPE == "matern" ]; then
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n $N --dimension $DIM \
        --lengthscale $LENSCALE --outputscale $KS --noise_variance $NV \
        --kernel_type "rbf" --nu $NU \
        --out $OUTFILE --id $ID
else
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n $N --dimension $DIM \
        --lengthscale $LENSCALE --outputscale $KS --noise_variance $NV \
        --kernel_type $KTYPE \
        --out $OUTFILE --id $ID
fi

conda deactivate
