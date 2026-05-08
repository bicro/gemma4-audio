# Data Notes

The step-500 smoke run trained on Qwen3-TTS-generated teacher audio, not on a
large real-world speech corpus.

## Full Run

The full run started from `656` candidate train rows and used `TRAIN_LIMIT=128`
for the smoke training run.

Included in `data/full_run/`:

- `train_raw_portable.jsonl`: full candidate train manifest.
- `train_with_mimi_codes_portable.jsonl`: the actual 128 training rows with Mimi
  token targets.
- `valid_with_mimi_codes_portable.jsonl`: validation rows with Mimi token
  targets.
- `test_with_mimi_codes_portable.jsonl`: held-out rows with Mimi token targets.
- `mimi_prepare_summary.json`: sample rate, frame rate, rows, and frame ranges.

Not included in git:

- the full teacher WAV directory
- the full Gemma hidden-state caches
- the trained step-500 checkpoint

These files are excluded to keep the repository lightweight.

## Mini Set

`data/mini/` includes a tiny subset for local pipeline checks:

- `8` train teacher WAVs
- `3` validation teacher WAVs
- `3` test teacher WAVs
- `qwen-default-reference.wav`
- raw manifests and precomputed Mimi-code manifests

These samples are enough to verify the dataset shape and training scripts. They
are not enough to train a generally useful model.

## Why Qwen3-TTS Teacher Audio

The experiment needed short, clean, aligned text/audio pairs with a consistent
voice. Qwen3-TTS was used as the teacher source for that smoke test. The model
under test is not Qwen: Gemma hidden states condition a trainable head that
predicts Mimi codec tokens.

## Audio-Input Derived Data

The audio-input run reused the same teacher rows and Mimi-code targets, but
changed the Gemma input side. Instead of passing text into Gemma, the cache step
passed audio through Gemma's multimodal processor.

For the no-text prompt-contained run, each input WAV was derived as:

```text
spoken instruction prefix + short silence + original teacher target WAV
```

The manifest builder stores those longer files in a new output directory and
keeps the original target WAV path in `source_audio`. The `mimi_codes` field is
not regenerated from the longer prompt-contained WAV; it remains the target
sentence audio code sequence. That is intentional: the adapter learns to map
the full spoken prompt context to the desired spoken response.

See `docs/AUDIO_INPUT_TRAINING.md` for the exact command.
