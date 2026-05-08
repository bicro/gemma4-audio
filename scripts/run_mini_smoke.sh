#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/mini}"
RUN_DIR="${2:-runs/mini}"

GEMMA_MODEL_PATH="${GEMMA_MODEL_PATH:-google/gemma-4-E4B-it}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_CODEBOOKS="${NUM_CODEBOOKS:-8}"
MAX_STEPS="${MAX_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1e-4}"

mkdir -p "${RUN_DIR}"

echo "[mini] encoding teacher WAVs to Mimi codes"
"${PYTHON_BIN}" scripts/prepare_mimi_dataset.py \
  --input_dir "${DATA_DIR}" \
  --output_dir "${RUN_DIR}/data" \
  --num_codebooks "${NUM_CODEBOOKS}" \
  --write_recon_samples 1

echo "[mini] caching frozen Gemma hidden states"
"${PYTHON_BIN}" scripts/cache_gemma_mimi_features.py \
  --input_jsonl "${RUN_DIR}/data/train_with_mimi_codes.jsonl" \
  --output "${RUN_DIR}/data/train_gemma_mimi_cache.pt" \
  --gemma_model_path "${GEMMA_MODEL_PATH}"

"${PYTHON_BIN}" scripts/cache_gemma_mimi_features.py \
  --input_jsonl "${RUN_DIR}/data/valid_with_mimi_codes.jsonl" \
  --output "${RUN_DIR}/data/valid_gemma_mimi_cache.pt" \
  --gemma_model_path "${GEMMA_MODEL_PATH}"

echo "[mini] training Gemma-to-Mimi head"
PYTHONPATH=. "${PYTHON_BIN}" scripts/train_gemma_mimi_head.py \
  --train_cache "${RUN_DIR}/data/train_gemma_mimi_cache.pt" \
  --valid_cache "${RUN_DIR}/data/valid_gemma_mimi_cache.pt" \
  --output_dir "${RUN_DIR}/train" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --max_steps "${MAX_STEPS}" \
  --save_every_steps "${MAX_STEPS}" \
  --num_codebooks "${NUM_CODEBOOKS}"

echo "[mini] done: ${RUN_DIR}"
