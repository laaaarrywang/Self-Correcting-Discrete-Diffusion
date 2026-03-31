#!/bin/bash

REPO_DIR="<REPO_DIR>"
STORAGE_BASE="<FAST_STORAGE_DIR>"
GIDD_EVAL_BASE="<EVAL_OUTPUT_DIR>"
LOG_DIR="${REPO_DIR}/gidd/logs"

mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/official_loss_$$.out" 2>&1

echo "=== GIDD Official Checkpoint Val PPL via eval_metrics[elbo] ==="
echo "Job ID: $$"
date

# HuggingFace cache
export HF_HOME=${GIDD_EVAL_BASE}/hf_cache
export HF_HUB_CACHE=${GIDD_EVAL_BASE}/hf_cache
export HF_DATASETS_CACHE=${STORAGE_BASE}/data
export TRANSFORMERS_CACHE=${GIDD_EVAL_BASE}/hf_cache
export HF_DATASETS_OFFLINE=1

# Load modules and activate environment
module load conda
conda activate gidd

cd "${REPO_DIR}/gidd"
export PYTHONPATH="${REPO_DIR}/gidd:${PYTHONPATH:-}"

BATCH_SIZE=${BATCH_SIZE:-16}

HF_MODELS=(
    "dvruette/gidd-small-p_unif-0.2"
    "dvruette/gidd-small-p_unif-0.1"
)

for HF_MODEL in "${HF_MODELS[@]}"; do
    MODEL_SLUG=$(echo "${HF_MODEL}" | tr '/' '-')
    METRICS_DIR="${GIDD_EVAL_BASE}/${MODEL_SLUG}/loss_ppl"
    METRICS_PATH="${METRICS_DIR}/loss_ppl.json"
    RUN_DIR="${METRICS_DIR}/run"
    RUN_LOG="${METRICS_DIR}/run.log"
    mkdir -p "${METRICS_DIR}"

    if [ -f "${METRICS_PATH}" ]; then
        echo "Skipping ${HF_MODEL}: already computed at ${METRICS_PATH}"
        continue
    fi

    echo ""
    echo "=== Computing loss PPL for: ${HF_MODEL} (${NUM_GPUS} GPU) ==="
    echo "  Output: ${METRICS_PATH}"

    rm -rf "${RUN_DIR}"
    mkdir -p "${RUN_DIR}"

    if ! CUDA_VISIBLE_DEVICES=0 python -u \
        "${REPO_DIR}/gidd/gidd/eval/loss.py" \
        path=dummy \
        batch_size=${BATCH_SIZE} \
        +hf_model=${HF_MODEL} \
        +hf_cache_dir=${GIDD_EVAL_BASE}/hf_cache \
        +cache_dir=${STORAGE_BASE}/gidd_cache \
        +data_dir=${STORAGE_BASE}/data \
        hydra.run.dir="${RUN_DIR}" \
        > "${RUN_LOG}" 2>&1; then
        echo "ERROR: evaluation failed for ${HF_MODEL}"
        tail -20 "${RUN_LOG}"
        continue
    fi

    if [ ! -f "${RUN_DIR}/metrics.json" ]; then
        echo "ERROR: missing ${RUN_DIR}/metrics.json for ${HF_MODEL}"
        tail -20 "${RUN_LOG}"
        continue
    fi

    cp "${RUN_DIR}/metrics.json" "${METRICS_PATH}"
    rm -rf "${RUN_DIR}"

    echo "Done: ${HF_MODEL}"
    echo ""
done

echo "=== All evaluations complete ==="
date
