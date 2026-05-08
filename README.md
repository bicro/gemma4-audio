# Gemma 4 E4B Audio Head Smoke Test

This repo packages a small research demo for mapping frozen Gemma 4 E4B hidden
states into Kyutai Mimi codec tokens.

The narrow claim is:

```text
text -> frozen Gemma hidden states -> trainable Mimi token head -> frozen Mimi decoder -> WAV
```

The newer audio-input smoke path uses the same head shape, but reads Gemma
states produced from audio input:

```text
spoken WAV -> frozen Gemma audio-conditioned hidden states -> trainable Mimi token head -> frozen Mimi decoder -> WAV
```

This is not a production TTS model. It is an overfit smoke test showing that a
Gemma-conditioned head can emit neural audio codec tokens that decode into
recognizable speech.

## What Is Included

- `gemma_kyutai_tts/`: the trainable head module.
- `scripts/`: dataset prep, Gemma feature caching, head training, generation,
  and ASR word-overlap evaluation.
- `scripts/generate_gemma_mimi_from_audio.py`: audio-input inference for the
  Gemma-to-Mimi head. Pass `--instruction ""` for the no-text-prompt path.
- `scripts/build_audio_prompt_contained_manifest.py`: creates a derived
  manifest where each Gemma input WAV contains a spoken instruction prefix plus
  target speech, while the training target remains the target Mimi codes.
- `scripts/run_kyutai_mimi_text_audio_until_asr.sh`: ASR-gated continuation
  runner for text+audio, audio-only, and no-text audio-input experiments.
- `data/mini/`: a tiny Qwen3-TTS-generated teacher set with audio files, Mimi
  token targets, and raw manifests.
- `data/full_run/`: text manifests and Mimi-token targets from the step-500 run.
  The full teacher WAV set, Gemma caches, and checkpoint are intentionally not
  committed to git.
- `artifacts/step500/`: generated step-500 WAVs, ASR logs, metadata, and the
  head config.

## Large Files Not Included

The trained step-500 checkpoint, full teacher WAV directory, and Gemma
hidden-state caches are not checked in. The repo includes the code, manifests,
Mimi token targets, generated samples, and logs needed to inspect the run and
rerun the pipeline locally.

## Install

Use a CUDA machine for Gemma feature extraction or generation. The mini smoke
script will create local Gemma hidden-state caches under `runs/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

You also need access to `google/gemma-4-E4B-it` on Hugging Face for cache and
generation commands.

The RunPod setup that worked for the Gemma4/Mimi path used:

```bash
pip install "torch==2.9.1" "transformers>=5.8,<5.9" "huggingface_hub>=1.5,<2"
pip install --no-deps "moshi==0.2.13"
pip install soundfile safetensors sentencepiece accelerate einops sphn "numpy<2"
```

Avoid importing the base-image `torchaudio` if it was built for a different
Torch version. The included ASR evaluator resamples with `soundfile`/NumPy so
Whisper evaluation does not need `torchaudio.functional`.

## Mini Smoke Run

This command encodes the mini teacher WAVs with Mimi, caches Gemma hidden
states, and trains a tiny local run. It is intended to check that the pipeline
is wired correctly, not to produce a good voice.

```bash
./scripts/run_mini_smoke.sh data/mini runs/mini
```

The script writes generated files under `runs/`, which is git-ignored.

## Generate From A Local Checkpoint

After running a local smoke or training job, point `--checkpoint` at the
checkpoint it wrote under `runs/`:

```bash
PYTHONPATH=. python scripts/generate_gemma_mimi.py \
  --checkpoint runs/mini/train/mimi-head-step-10.pt \
  --output_dir verify_samples \
  --gemma_model_path google/gemma-4-E4B-it \
  --seconds 3.6 \
  --num_codebooks 8 \
  --temperature 0.0 \
  --text "the final layer generates audio codes for the demo."
```

That command passes text only to Gemma. The Mimi stage receives predicted codec
tokens, not text.

## Generate From Audio Input

The audio-input script lets Gemma consume a WAV and lets the trained head read
Gemma's audio-conditioned hidden states. To test the strict no-text path, pass
an empty instruction:

```bash
PYTHONPATH=. python scripts/generate_gemma_mimi_from_audio.py \
  --checkpoint runs/audio_prompt/train/mimi-head-step-850.pt \
  --output_dir verify_audio_input \
  --gemma_model_path google/gemma-4-E4B-it \
  --audio /path/to/audio_prompt_contained_input.wav \
  --label "the small adapter conditions clear speech through frozen layers." \
  --seconds 3.68 \
  --num_codebooks 8 \
  --temperature 0.0 \
  --instruction ""
```

With `--instruction ""`, the Gemma chat content contains only the audio object.
The prompt must be spoken inside the WAV if you want an instruction. The smoke
run used a WAV that said:

```text
Repeat the following sentence naturally as speech.
The small adapter conditions clear speech through frozen layers.
```

The step-850 checkpoint ASR output was:

```text
The small adapter conditions clear speeds through frozen mirrors.
```

That matched six target words: `small`, `adapter`, `conditions`, `clear`,
`through`, and `frozen`.

## Training The Audio-Input Path

`cache_gemma_mimi_features.py` supports three cache modes:

```bash
--input_mode text   # text prompt -> Gemma hidden states
--input_mode audio  # audio prompt -> Gemma hidden states
--input_mode both   # two examples per row, same Mimi target codes
```

For no-text audio input, use:

```bash
export CACHE_INPUT_MODE=audio
export AUDIO_PROMPT_TEMPLATE=""
```

For prompt-contained audio training, build a derived dataset first:

```bash
python scripts/build_audio_prompt_contained_manifest.py \
  --input_dir /workspace/data/qwen3_tts_default_voice_teacher \
  --output_dir /workspace/data/qwen3_tts_default_voice_teacher_audio_prompt_contained \
  --prefix_audio /workspace/inputs/repeat_instruction_prefix_16k.wav \
  --splits train,valid,test \
  --gap_sec 0.25
```

Then continue adapter-only training from a compatible checkpoint:

```bash
INITIAL_CHECKPOINT=/workspace/runs/gemma_kyutai_mimi_audio_only_no_text_trained_20260508_2230/train/mimi-head-step-800.pt \
CACHE_INPUT_MODE=audio \
AUDIO_PROMPT_TEMPLATE="" \
TARGET_STEPS="850 900 1000 1250 1500 2000" \
EVAL_AUDIO=/workspace/inputs/audio_prompt_prefix_plus_qwen_blog_sample_03.wav \
EVAL_SECONDS=3.68 \
EVAL_LABEL="the small adapter conditions clear speech through frozen layers." \
scripts/run_kyutai_mimi_text_audio_until_asr.sh \
  /workspace/data/qwen3_tts_default_voice_teacher_audio_prompt_contained \
  /workspace/runs/gemma_kyutai_mimi_audio_prompt_contained
```

The runner freezes Gemma and Mimi. It trains only `GemmaToMimiCodecHead` with
weighted cross-entropy over Mimi codebook IDs, then runs Whisper word-overlap
checks at each target checkpoint.

For the full audio-input run notes, including the passing checkpoint and exact
ASR output, see [`docs/AUDIO_INPUT_TRAINING.md`](docs/AUDIO_INPUT_TRAINING.md).

## Caveats

- The blog/demo samples are training-set overfit samples, not held-out proof.
- The step-500 run trained on 128 short Qwen3-TTS teacher clips.
- The audio-input step-850 run is also a narrow smoke test. It proves the
  wiring can work for a controlled prompt-contained WAV, not general
  speech-to-speech instruction following.
- The current head predicts a fixed/parallel sequence of Mimi tokens. It is not
  an autoregressive speech decoder with robust duration control.
- The current scripts do not make Gemma autoregressively answer a question and
  then speak that answer. They render content represented in Gemma's input or
  audio-conditioned hidden states.
- The ASR gate is intentionally weak: simple word overlap, not WER or human MOS.
- Gemma, Qwen3-TTS, and Kyutai Mimi are external models with their own terms.
  This repo does not redistribute Gemma weights.
