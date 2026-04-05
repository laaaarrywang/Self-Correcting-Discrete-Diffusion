#!/bin/bash
set -euo pipefail

echo "=== SCDD vs GIDD Generation Latency Benchmark ==="
date

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="<REPO_DIR>"
DATA_DIR="<DATA_DIR>"

# ── Checkpoints ───────────────────────────────────────────────────────────────
SCDD_CKPT="${DATA_DIR}/scdd_checkpoints/scdd.ckpt"
GIDD_CKPT="${DATA_DIR}/gidd_checkpoints/latest"

# ── Benchmark parameters ─────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-512}"
NUM_WARMUP="${NUM_WARMUP:-3}"
NUM_TIMED="${NUM_TIMED:-10}"
STEPS="${STEPS:-32,64,128,256,512}"
OUTPUT_DIR="${DATA_DIR}/bench_latency"
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "Configuration:"
echo "  SCDD_CKPT  : ${SCDD_CKPT}"
echo "  GIDD_CKPT  : ${GIDD_CKPT}"
echo "  BATCH_SIZE  : ${BATCH_SIZE}"
echo "  SEQ_LEN     : ${SEQ_LEN}"
echo "  NUM_WARMUP  : ${NUM_WARMUP}"
echo "  NUM_TIMED   : ${NUM_TIMED}"
echo "  STEPS       : ${STEPS}"
echo "  OUTPUT_DIR  : ${OUTPUT_DIR}"
echo ""

# Run both benchmarks sequentially on the SAME GPU for fair comparison
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "  SCDD Benchmark"
echo "============================================================"
python -u ${REPO_DIR}/rebuttal/bench_latency_scdd.py \
    --checkpoint_path "${SCDD_CKPT}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --num_warmup "${NUM_WARMUP}" \
    --num_timed "${NUM_TIMED}" \
    --steps "${STEPS}" \
    --output_path "${OUTPUT_DIR}/bench_scdd.json"

echo ""
echo "============================================================"
echo "  GIDD Benchmark"
echo "============================================================"
# Activate GIDD conda environment if different, e.g.:
# conda activate gidd
python -u ${REPO_DIR}/rebuttal/bench_latency_gidd.py \
    --checkpoint_path "${GIDD_CKPT}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --num_warmup "${NUM_WARMUP}" \
    --num_timed "${NUM_TIMED}" \
    --steps "${STEPS}" \
    --output_path "${OUTPUT_DIR}/bench_gidd.json"

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
python3 -c "
import json

with open('${OUTPUT_DIR}/bench_scdd.json') as f:
    scdd = json.load(f)
with open('${OUTPUT_DIR}/bench_gidd.json') as f:
    gidd = json.load(f)

print(f'GPU: {scdd[\"gpu\"]}')
print(f'Batch size: {scdd[\"batch_size\"]}, Seq len: {scdd[\"seq_len\"]}')
print()
print(f'{\"Steps\":>6}  {\"SCDD (s)\":>14}  {\"GIDD (s)\":>14}  {\"SCDD tok/s\":>14}  {\"GIDD tok/s\":>14}  {\"Ratio\":>8}')
print('-' * 80)
for s, g in zip(scdd['benchmarks'], gidd['benchmarks']):
    assert s['num_steps'] == g['num_steps']
    ns = s['num_steps']
    sl = s['mean_latency']
    gl = g['mean_latency']
    st = s['throughput_tok_per_sec']
    gt = g['throughput_tok_per_sec']
    ratio = gl / sl
    print(f'{ns:>6}  {sl:>10.3f} ± {s[\"std_latency\"]:.3f}  {gl:>10.3f} ± {g[\"std_latency\"]:.3f}  {st:>14.0f}  {gt:>14.0f}  {ratio:>7.2f}x')
"

echo ""
echo "=== Benchmark complete ==="
date
