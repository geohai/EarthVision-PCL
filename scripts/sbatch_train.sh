#!/bin/bash
#SBATCH --job-name=loc4pm_train
#SBATCH --account=cis250170p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-31:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=experiments/logs/%x_%j.out

source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate  /ocean/projects/cis250170p/zwang63/envs/pt-cu121

cd /ocean/projects/cis250170p/zwang63/src/loc4pm
export PYTHONPATH=$(pwd):$PYTHONPATH

python -m loc4pm.utils.print_env

# example overrides:
# python -m loc4pm.train --config configs/default.yaml split.name spatial split.spatial.fold_index 3

python -m loc4pm.train --config configs/default.yaml