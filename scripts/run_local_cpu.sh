#!/usr/bin/env bash
set -euo pipefail
source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250170p/zwang63/envs/pt-cu121
cd /ocean/projects/cis250170p/zwang63/src/loc4pm
export PYTHONPATH=$(pwd):$PYTHONPATH

# install deps once (safe to re-run)
#pip install -r requirements.txt

python -m loc4pm.utils.print_env

# optional tests
pytest -q || true

# tiny CPU smoke run (adjust dates to files you have)
python -m loc4pm.train --config configs/default.yaml   data.start_date 2018-01-01 data.end_date 2018-01-03   train.device cpu train.epochs 1 train.batch_size 16   data.num_workers 0 logging.tensorboard false   split.name random split.seed 123 split.test_split 0.1 split.val_split 0.1