# External Models

This repo does not redistribute Gemma, Qwen3-TTS, or Kyutai model weights.

## Gemma 4 E4B

- Model used for hidden states: `google/gemma-4-E4B-it`
- Role in this experiment: frozen language model backbone.
- Users must accept Google's Gemma terms on Hugging Face before running cache or
  generation commands.

## Kyutai Mimi

- Source: `kyutai/moshiko-pytorch-bf16`
- Weight used by the smoke run: `tokenizer-e351c8d8-checkpoint125.safetensors`
- Role in this experiment: frozen audio codec encoder/decoder.

## Qwen3-TTS

- Role in this experiment: teacher audio source for short, clean, aligned
  prompt/audio examples.
- Qwen is not used as the model under test. The trained component is a
  Gemma-conditioned Mimi-token head.

Before publishing, confirm the current upstream licenses/terms and add any
required attribution to the GitHub release page.
