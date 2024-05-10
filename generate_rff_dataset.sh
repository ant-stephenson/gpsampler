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
#SBATCH --array=1
#SBATCH --mail-user=$USER@bristol.ac.uk
#SBATCH --mail-type=END
#SBATCH --output=/user/work/ll20823/mini-project/slurm/sampling/rff-%j_%a.out


SCRIPT_DIR=/user/work/ll20823/mini-project

module load lang/python/anaconda/3.8.5-2021-AM

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.8.5-2021-AM/lib

source $SCRIPT_DIR/projenv/bin/activate

echo $SLURM_JOB_START_TIME

DIM=100
LENSCALE=3.0
KTYPE="exp"

if [ $KTYPE == "exp" ]; then
  python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n 100000 --dimension $DIM \
    --lengthscale $LENSCALE --outputscale 0.9 --noise_variance 0.1 \
    --kernel_type "matern" --nu 0.5 \
    --out "${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KTYPE}_dim_${DIM}_ls_${LENSCALE}.npy" 
elif [ $KTYPE == "rbf" ]; then
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n 100000 --dimension $DIM \
        --lengthscale $LENSCALE --outputscale 0.9 --noise_variance 0.1 --kernel_type "rbf" \
        --out "${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KTYPE}_dim_${DIM}_ls_${LENSCALE}.npy" 
else
    python $SCRIPT_DIR/gpsampler/generate_rff_dataset.py --n 100000 --dimension $DIM \
        --lengthscale $LENSCALE --outputscale 0.9 --noise_variance 0.1 \
        --kernel_type "matern" --nu $NU \
        --out "${SCRIPT_DIR}/synthetic-datasets/RFF/output_kt_${KTYPE}_dim_${DIM}_ls_${LENSCALE}.npy" 
fi

deactivate
