#!/bin/bash
#SBATCH --job-name=loc4pm_train
#SBATCH --account=cis250170p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err

set -eox pipefail
mkdir -p experiments/logs

set +u
source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate  /ocean/projects/cis250170p/zwang63/envs/pt-cu121
set -u

cd /ocean/projects/cis250170p/zwang63/src/loc4pm
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

# 1) Stage data to node-local scratch
SRC=/ocean/projects/cis250170p/zwang63/data/loc4pm/processed/EO_NCAR_merged
DST=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/data
mkdir -p "$DST"
rsync -a --info=progress2 "$SRC"/ "$DST"/

nvidia-smi
python -m loc4pm.utils.print_env

# example overrides:
# python -m loc4pm.train --config configs/default.yaml split.name spatial split.spatial.fold_index 3

python -m loc4pm.train --config configs/default.yaml \
  data.root "$DST"