#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
DATA=data/msd/npz
SREP=data/msd/srep_work/srep
MESH=data/msd/srep_work/mesh
for FOLD in 0 1 2 3 4; do
    RUN=runs/msd_fold${FOLD}
    CUDA_VISIBLE_DEVICES=$GPU python train.py --data_dir $DATA --srep_dir $SREP --fold $FOLD --logdir $RUN
    CUDA_VISIBLE_DEVICES=$GPU python test.py --data_dir $DATA --srep_dir $SREP --fold $FOLD \
        --checkpoint $RUN/model_best_angle.pt --out $RUN/test --mesh_dir $MESH
done
python - <<'PY'
import csv, numpy as np
rows = [next(csv.DictReader(open(f"runs/msd_fold{k}/test/metrics.csv"))) for k in range(5)]
for key in [k for k in rows[0] if k != "case"]:
    v = np.array([float(r[key]) for r in rows])
    print(f"{key:16s} {v.mean():.4f} ± {v.std():.4f}  (folds: {' '.join(f'{x:.4f}' for x in v)})")
PY
