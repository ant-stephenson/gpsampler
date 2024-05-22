#!/bin/bash

#SBATCH --job-name=sample_ciq
#SBATCH --partition=compute
#SBATCH --account=math026082 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --array=0
#SBATCH --mail-user=$USER@bristol.ac.uk
#SBATCH --mail-type=END
#SBATCH --output=/user/work/ll20823/mini-project/slurm/sampling/ciq-%j_%a.out


source slurm_init.sh

DIMS=(10 50 100)
DIMS+=( "${DIMS[@]}" "${DIMS[@]}" )
LENSCALES=(4.0 4.0 4.0 5.0 5.0 5.0 6.0 6.0 6.0)

DIM="${DIMS[$SLURM_ARRAY_TASK_ID]}"
LENSCALE="${LENSCALES[$SLURM_ARRAY_TASK_ID]}"
KTYPE="exp"

ID=1

OUTFILE="${SCRIPT_DIR}/synthetic-datasets/CIQ/output_kt_${KTYPE}_dim_${DIM}_ls_${LENSCALE}_${ID}.npy" 

if [ $KTYPE == "exp" ]; then
    python $SCRIPT_DIR/gpsampler/generate_ciq_dataset.py --n 100000 --dimension $DIM \
        --lengthscale $LENSCALE --outputscale 0.9 --noise_variance 0.1 \
        --kernel_type "exp" --nu 0.5 \
        --out $OUTFILE
elif [ $KTYPE == "rbf" ]; then
    python $SCRIPT_DIR/gpsampler/generate_ciq_dataset.py --n 100000 --dimension $DIM \
        --lengthscale $LENSCALE --outputscale 0.9 --noise_variance 0.1 --kernel_type "rbf" \
        --out $OUTFILE
else
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n 100000 --dimension $DIM \
        --lengthscale $LENSCALE --outputscale 0.9 --noise_variance 0.1 \
        --kernel_type $KTYPE \
        --out $OUTFILE
fi

conda deactivate
