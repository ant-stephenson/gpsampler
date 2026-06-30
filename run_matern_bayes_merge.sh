#!/bin/bash

#SBATCH --job-name=matern_bayes_merge
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

SCRIPT_DIR=/user/work/ll20823/gpsampler
OUTDIR=$SCRIPT_DIR/sweeps/matern_bayes/output

source $SCRIPT_DIR/projenv/bin/activate

module load lang/python/anaconda/3.8.5-2021-AM

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.8.5-2021-AM/lib

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
