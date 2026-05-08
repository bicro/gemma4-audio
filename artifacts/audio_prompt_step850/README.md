# Audio-Prompt Step-850 Sample

This directory contains the audio-in/audio-out sample used by the blog post
`Grafting a Speech Head onto Gemma 4 E4B`.

The checkpoint was:

```text
/workspace/runs/gemma_kyutai_mimi_audio_prompt_contained_trained_20260508_2245/train/mimi-head-step-850.pt
```

Included:

- `generated_samples/audio_prompt_contained_input.wav`: the audio input passed
  to Gemma. It contains the spoken instruction and spoken target sentence.
- `generated_samples/audio_prompt_contained_adapter_output.wav`: the
  Gemma-audio-states-to-Mimi output from the trained head.
- `metadata.json`: generation metadata. The key fields are
  `"input_mode": "audio_only_no_text_instruction"` and `"instruction": ""`.
- `asr.json`: Whisper word-overlap evaluation.

Not included:

- the step-850 checkpoint
- full Gemma caches
- the derived prompt-contained teacher WAV dataset

Input WAV content:

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
