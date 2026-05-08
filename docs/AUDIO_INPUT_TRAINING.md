# Audio-Input Training Notes

This note captures the changes used for the Gemma audio-input to Mimi
audio-output smoke test.

## What Changed

The original demo trained:

```text
text prompt -> frozen Gemma hidden states -> GemmaToMimiCodecHead -> Mimi codes
```

The audio-input demo trains the same head on hidden states from Gemma's audio
path:

```text
WAV prompt -> frozen Gemma audio-conditioned hidden states -> GemmaToMimiCodecHead -> Mimi codes
```

Gemma is frozen. Kyutai Mimi is frozen. The only trainable module is still
`GemmaToMimiCodecHead`.

## Cache Modes

`scripts/cache_gemma_mimi_features.py` now accepts:

```bash
--input_mode text
--input_mode audio
--input_mode both
```

For `audio`, the script loads `row["audio"]`, passes it through Gemma's
multimodal processor, and caches the last Gemma decoder layers. For `both`, it
creates one text-conditioned item and one audio-conditioned item for each row,
with the same target Mimi codes.

Use `--audio_prompt_template ""` for the strict no-text path. That produces
Gemma messages with only an audio object in the user content.

## Prompt-Contained Audio Dataset

For the stricter demo, the instruction was spoken inside the input WAV instead
of being passed as text. The input WAV format was:

```text
[spoken instruction prefix]
0.25 seconds silence
[spoken target sentence]
```

The target Mimi codes stayed aligned to only the target sentence, not the
instruction prefix.

Build this derived manifest with:

```bash
python scripts/build_audio_prompt_contained_manifest.py \
  --input_dir /workspace/data/qwen3_tts_default_voice_teacher \
  --output_dir /workspace/data/qwen3_tts_default_voice_teacher_audio_prompt_contained \
  --prefix_audio /workspace/inputs/repeat_instruction_prefix_16k.wav \
  --splits train,valid,test \
  --gap_sec 0.25
```

The script rewrites `audio` to the longer prompt-contained WAV and stores the
original teacher WAV in `source_audio`. It preserves `mimi_codes`, so the loss
still trains the adapter to output only the target sentence.

## ASR-Gated Run

The passing run used this command shape on RunPod:

```bash
INITIAL_CHECKPOINT=/workspace/runs/gemma_kyutai_mimi_audio_only_no_text_trained_20260508_2230/train/mimi-head-step-800.pt \
CACHE_INPUT_MODE=audio \
AUDIO_PROMPT_TEMPLATE="" \
TARGET_STEPS="850 900 1000 1250 1500 2000" \
SAVE_EVERY_STEPS=50 \
LR=1e-4 \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
EVAL_AUDIO=/workspace/inputs/audio_prompt_prefix_plus_qwen_blog_sample_03.wav \
EVAL_SECONDS=3.68 \
EVAL_LABEL="the small adapter conditions clear speech through frozen layers." \
scripts/run_kyutai_mimi_text_audio_until_asr.sh \
  /workspace/data/qwen3_tts_default_voice_teacher_audio_prompt_contained \
  /workspace/runs/gemma_kyutai_mimi_audio_prompt_contained_trained_20260508_2245
```

The runner:

1. Builds audio-conditioned Gemma caches.
2. Resumes the head from `INITIAL_CHECKPOINT`.
3. Trains only the adapter with weighted per-codebook cross entropy.
4. Generates audio at each target checkpoint.
5. Runs Whisper word-overlap evaluation.
6. Stops when the ASR gate passes.

## Passing Result

Checkpoint:

```text
/workspace/runs/gemma_kyutai_mimi_audio_prompt_contained_trained_20260508_2245/train/mimi-head-step-850.pt
```

Gemma input:

```text
no text instruction
audio object only
```

The WAV said:

```text
Repeat the following sentence naturally as speech.
The small adapter conditions clear speech through frozen layers.
```

Whisper transcript of the adapter output:

```text
The small adapter conditions clear speeds through frozen mirrors.
```

Matched target words:

```text
small, adapter, conditions, clear, through, frozen
```

The blog sample artifacts are checked in at:

```text
artifacts/audio_prompt_step850/
```

## Important Caveats

This is not broad speech-to-speech instruction following. It is a controlled
architecture smoke test showing that Gemma audio-conditioned hidden states can
drive a frozen Mimi decoder through a small trained head.

The current inference scripts do not run Gemma's autoregressive answer
generation loop and then speak the generated answer. They render content that
is represented in the prompt/audio hidden states.
