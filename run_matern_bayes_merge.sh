#!/bin/bash

#SBATCH --job-name=matern_bayes_merge
#SBATCH --output=/user/work/ll20823/mini-project/slurm/matern_bayes_merge/%x.%j.out
#SBATCH --error=/user/work/ll20823/mini-project/slurm/matern_bayes_merge/%x.%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --mem-per-cpu=4G

# ---------------------------------------------------------------------------
# Merge per-method CSVs from run_matern_bayes.sh into a single file for
# the figure scripts.
#
# Submit after the array completes:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> run_matern_bayes_merge.sh
#
# Optional: set D and TAG to match the array job (defaults: D=1, TAG=merged)
# ---------------------------------------------------------------------------

SCRIPT_DIR=/user/work/ll20823/mini-project/gpsampler
OUTDIR=$SCRIPT_DIR/sweeps/matern_bayes/output

module load languages/python/3.8.20

source $SCRIPT_DIR/.ve38/bin/activate

D=${D:-1}
TAG=${TAG:-merged}

python - <<EOF
import pathlib, pandas as pd, sys

outdir = pathlib.Path("$OUTDIR")
d = $D
tag = "$TAG"

# Find the per-method CSVs produced by the array job
pattern = f"matern_bayes_d{d}_*_d{d}_*.csv"
csvs = sorted(outdir.glob(pattern))
if not csvs:
    sys.exit(f"No CSVs found in {outdir} matching {pattern}")

print(f"Merging {len(csvs)} files:")
for p in csvs:
    print(f"  {p.name}")

df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
df = df.sort_values(["method", "n", "nu", "ell", "fidelity"]).reset_index(drop=True)

out = outdir / f"matern_bayes_d{d}_{tag}.csv"
df.to_csv(out, index=False)
print(f"\nMerged {len(df)} rows -> {out}")
EOF

deactivate
