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
#SBATCH --array=0-8
#SBATCH --mail-user=$USER@bristol.ac.uk
#SBATCH --mail-type=END
#SBATCH --output=/user/work/ll20823/mini-project/slurm/sampling/rff-%j_%a.out


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
KTYPE="exp"

ID=1

# already done this one and they take ages...
# if [ $DIM == 100 ] && [ $LENSCALE == 3.0 ] && [ $KTYPE == "exp" ]; then
#     exit 1
# fi

OUTFILE="${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KTYPE}_dim_${DIM}_ls_${LENSCALE}_${ID}.npy" 

if [ $KTYPE == "exp" ]; then
  python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n 100000 --dimension $DIM \
    --lengthscale $LENSCALE --outputscale 0.9 --noise_variance 0.1 \
    --kernel_type "matern" --nu 0.5 \
    --out $OUTFILE
elif [ $KTYPE == "rbf" ]; then
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n 100000 --dimension $DIM \
        --lengthscale $LENSCALE --outputscale 0.9 --noise_variance 0.1 --kernel_type "rbf" \
        --out $OUTFILE
else
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n 100000 --dimension $DIM \
        --lengthscale $LENSCALE --outputscale 0.9 --noise_variance 0.1 \
        --kernel_type $KTYPE \
        --out $OUTFILE
fi

deactivate
