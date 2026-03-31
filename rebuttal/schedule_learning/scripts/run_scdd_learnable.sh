#!/usr/bin/env bash

set -euo pipefail

# Generic launcher for SCDD with the learnable corruption schedule.

REPO_DIR="<REPO_DIR>"
BASE_CONFIG="<BASE_CONFIG>"
DATA_CACHE_DIR="<DATA_CACHE_DIR>"
OUTPUT_DIR="<OUTPUT_DIR>"

MAIN_LR="${MAIN_LR:-5e-4}"
SCHEDULE_LR="${SCHEDULE_LR:-1e-5}"
RATIO="${RATIO:-0.1}"
T_PEAK="${T_PEAK:-0.5}"
DEVICES="${DEVICES:-1}"

cd "${REPO_DIR}/scdd"

python -u -m main \
  --config-name "${BASE_CONFIG}" \
  parameterization=scdd \
  noise=loglinear \
  forward=mix \
  forward.ratio="${RATIO}" \
  forward.t_peak="${T_PEAK}" \
  forward.schedule_version=learnable \
  optim.lr="${MAIN_LR}" \
  optim.schedule_lr="${SCHEDULE_LR}" \
  data.cache_dir="${DATA_CACHE_DIR}" \
  hydra.run.dir="${OUTPUT_DIR}/outputs" \
  checkpointing.save_dir="${OUTPUT_DIR}/checkpoints" \
  trainer.devices="${DEVICES}"
