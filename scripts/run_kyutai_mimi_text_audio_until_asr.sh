#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:?usage: run_kyutai_mimi_text_audio_until_asr.sh DATA_DIR RUN_DIR}"
RUN_DIR="${2:?usage: run_kyutai_mimi_text_audio_until_asr.sh DATA_DIR RUN_DIR}"

GEMMA_MODEL_PATH="${GEMMA_MODEL_PATH:-google/gemma-4-E4B-it}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/gemma_kyutai_mimi}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-/workspace/checkpoints/mimi-head-step-500.pt}"
TARGET_STEPS="${TARGET_STEPS:-600 750 1000 1500 2000 3000}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-5}"
NUM_CODEBOOKS="${NUM_CODEBOOKS:-8}"
CACHE_INPUT_MODE="${CACHE_INPUT_MODE:-both}"
EVAL_SECONDS="${EVAL_SECONDS:-3.68}"
EVAL_AUDIO="${EVAL_AUDIO:-/workspace/inputs/qwen_tts_blog_sample_03.wav}"
EVAL_LABEL="${EVAL_LABEL:-the small adapter conditions clear speech through frozen layers.}"
ASR_MODEL="${ASR_MODEL:-openai/whisper-small.en}"
MIN_CORRECT_WORDS="${MIN_CORRECT_WORDS:-2}"
MIN_PASSING_SAMPLES="${MIN_PASSING_SAMPLES:-1}"
AUDIO_PROMPT_TEMPLATE="${AUDIO_PROMPT_TEMPLATE-Listen to the attached audio and repeat the spoken sentence naturally as speech.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${RUN_DIR}/train" "${RUN_DIR}/cache" "${RUN_DIR}/logs"

TRAIN_CACHE="${RUN_DIR}/cache/train_gemma_mimi_text_audio_cache.pt"
VALID_CACHE="${RUN_DIR}/cache/valid_gemma_mimi_text_audio_cache.pt"

latest_checkpoint() {
  find "${RUN_DIR}/train" -maxdepth 1 -name 'mimi-head-step-*.pt' 2>/dev/null \
    | sed -E 's/.*mimi-head-step-([0-9]+)\.pt/\1 &/' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

echo "[gemma-mimi-audio] run dir: ${RUN_DIR}"
echo "[gemma-mimi-audio] data dir: ${DATA_DIR}"
echo "[gemma-mimi-audio] building ${CACHE_INPUT_MODE} Gemma caches"

if [[ ! -f "${TRAIN_CACHE}" ]]; then
  PYTHONPATH="${EXPERIMENT_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/cache_gemma_mimi_features.py" \
    --input_jsonl "${DATA_DIR}/train_with_mimi_codes.jsonl" \
    --output "${TRAIN_CACHE}" \
    --gemma_model_path "${GEMMA_MODEL_PATH}" \
    --input_mode "${CACHE_INPUT_MODE}" \
    --audio_prompt_template "${AUDIO_PROMPT_TEMPLATE}" \
    > "${RUN_DIR}/logs/cache_train.log" 2>&1
fi

if [[ ! -f "${VALID_CACHE}" ]]; then
  PYTHONPATH="${EXPERIMENT_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/cache_gemma_mimi_features.py" \
    --input_jsonl "${DATA_DIR}/valid_with_mimi_codes.jsonl" \
    --output "${VALID_CACHE}" \
    --gemma_model_path "${GEMMA_MODEL_PATH}" \
    --input_mode "${CACHE_INPUT_MODE}" \
    --audio_prompt_template "${AUDIO_PROMPT_TEMPLATE}" \
    > "${RUN_DIR}/logs/cache_valid.log" 2>&1
fi

for target_step in ${TARGET_STEPS}; do
  current_checkpoint="$(latest_checkpoint || true)"
  if [[ -z "${current_checkpoint}" && -f "${INITIAL_CHECKPOINT}" ]]; then
    current_checkpoint="${INITIAL_CHECKPOINT}"
  fi
  resume_args=()
  if [[ -n "${current_checkpoint}" ]]; then
    resume_args=(--resume_checkpoint "${current_checkpoint}")
  fi

  echo "[gemma-mimi-audio] training until global step ${target_step}"
  PYTHONPATH="${EXPERIMENT_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_gemma_mimi_head.py" \
    --train_cache "${TRAIN_CACHE}" \
    --valid_cache "${VALID_CACHE}" \
    --output_dir "${RUN_DIR}/train" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --max_steps "${target_step}" \
    --save_every_steps "${SAVE_EVERY_STEPS}" \
    --num_codebooks "${NUM_CODEBOOKS}" \
    "${resume_args[@]}" \
    > "${RUN_DIR}/logs/train_to_${target_step}.log" 2>&1

  checkpoint="${RUN_DIR}/train/mimi-head-step-${target_step}.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    checkpoint="$(latest_checkpoint)"
  fi

  eval_dir="${RUN_DIR}/eval_audio_step_${target_step}"
  mkdir -p "${eval_dir}"
  echo "[gemma-mimi-audio] evaluating audio input at ${checkpoint}"

  PYTHONPATH="${EXPERIMENT_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_gemma_mimi_from_audio.py" \
    --checkpoint "${checkpoint}" \
    --output_dir "${eval_dir}/samples" \
    --gemma_model_path "${GEMMA_MODEL_PATH}" \
    --audio "${EVAL_AUDIO}" \
    --label "${EVAL_LABEL}" \
    --seconds "${EVAL_SECONDS}" \
    --num_codebooks "${NUM_CODEBOOKS}" \
    --temperature 0.0 \
    --instruction "${AUDIO_PROMPT_TEMPLATE}" \
    > "${eval_dir}/generate_audio.log" 2>&1

  "${PYTHON_BIN}" - <<PY
from pathlib import Path
import json
import numpy as np
import soundfile as sf

eval_dir = Path("${eval_dir}")
metadata_path = eval_dir / "samples" / "metadata.json"
metadata = json.loads(metadata_path.read_text())
for sample in metadata["samples"]:
    wav_path = Path(sample["adapter_wav"])
    wav, sr = sf.read(wav_path, dtype="float32")
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        norm_path = wav_path.with_name(wav_path.stem + "_peak_norm.wav")
        sf.write(norm_path, wav * (0.85 / peak), sr)
        sample["adapter_wav"] = str(norm_path)
        sample["raw_adapter_wav"] = str(wav_path)
        sample["normalization_peak_before"] = peak
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\\n")
PY

  set +e
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_asr_word_overlap.py" \
    --metadata "${eval_dir}/samples/metadata.json" \
    --output "${eval_dir}/asr.json" \
    --model "${ASR_MODEL}" \
    --min_correct_words "${MIN_CORRECT_WORDS}" \
    --min_passing_samples "${MIN_PASSING_SAMPLES}" \
    > "${eval_dir}/asr.log" 2>&1
  asr_status=$?
  set -e

  cat "${eval_dir}/asr.json" 2>/dev/null || tail -80 "${eval_dir}/asr.log"
  if [[ "${asr_status}" == "0" ]]; then
    echo "[gemma-mimi-audio] AUDIO PASS at ${checkpoint}"
    echo "${checkpoint}" > "${RUN_DIR}/PASS_AUDIO_CHECKPOINT.txt"
    echo "${eval_dir}" > "${RUN_DIR}/PASS_AUDIO_EVAL_DIR.txt"
    exit 0
  fi
done

echo "[gemma-mimi-audio] no ASR pass after target steps: ${TARGET_STEPS}" >&2
exit 1
