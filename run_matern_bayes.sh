#!/bin/bash

#SBATCH --job-name=matern_bayes
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=1-5

# ---------------------------------------------------------------------------
# Matérn Bayes-decision comparison — Stage 1 sweep (HPC, array over methods)
#
# Parallelises over the four methods via a SLURM array:
#   task 1 → rff
#   task 2 → lrff
#   task 3 → ciq      (slowest: O(n² J) per config at n=2048)
#   task 4 → pciq
#
# Each task writes its own CSV to OUTDIR.  After all four complete, run
# the merge job to produce a single file for the figure scripts:
#
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> run_matern_bayes_merge.sh
#
# To run the d=2 robustness sweep instead:
#   sbatch --export=ALL,D=2 run_matern_bayes.sh
# ---------------------------------------------------------------------------

SCRIPT_DIR=/user/work/ll20823/gpsampler
OUTDIR=$SCRIPT_DIR/sweeps/matern_bayes/output

source $SCRIPT_DIR/projenv/bin/activate

module load lang/python/anaconda/3.8.5-2021-AM

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.8.5-2021-AM/lib

# Map array task ID → method (1-indexed)
METHODS=("" "rff" "lrff" "ciq" "pciq" "elrff")
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

# Input dimension: default 1; override with --export=ALL,D=2
D=${D:-1}

# dtype and chunk_size: control RFF memory usage
#   DTYPE=float32 halves per-chunk intermediate memory (cos/sin arrays)
#   CHUNK_SIZE controls frequencies processed per chunk (default 512)
# Example: sbatch --export=ALL,D=1,DTYPE=float32,CHUNK_SIZE=1024 run_matern_bayes.sh
DTYPE=${DTYPE:-float64}
CHUNK_SIZE=${CHUNK_SIZE:-512}

echo "Task $SLURM_ARRAY_TASK_ID: method=$METHOD  d=$D  dtype=$DTYPE  chunk_size=$CHUNK_SIZE"
date

python -m sweeps.matern_bayes.run_sweep \
    --methods "$METHOD" \
    --d "$D" \
    --seed 42 \
    --outdir "$OUTDIR" \
    --tag "d${D}_${METHOD}" \
    --dtype "$DTYPE" \
    --chunk_size "$CHUNK_SIZE"

echo "Done: method=$METHOD  d=$D"
date

deactivate
