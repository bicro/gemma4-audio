#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR-check generated Gemma->Mimi WAVs for target word overlap.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--model", default="openai/whisper-small.en")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_correct_words", type=int, default=2)
    parser.add_argument("--min_passing_samples", type=int, default=1)
    parser.add_argument("--adapter_only", action="store_true", default=True)
    return parser.parse_args()


def words(text: str) -> list[str]:
    stop = {
        "a",
        "an",
        "and",
        "as",
        "for",
        "in",
        "into",
        "is",
        "it",
        "now",
        "of",
        "the",
        "this",
        "to",
    }
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in stop]


def read_audio(path: Path) -> dict:
    wav, sample_rate = sf.read(path)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sample_rate != 16000:
        target_len = max(1, int(round(wav.shape[0] * 16000 / sample_rate)))
        source_x = np.linspace(0.0, 1.0, num=wav.shape[0], endpoint=False)
        target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        wav = np.interp(target_x, source_x, wav).astype(np.float32)
        sample_rate = 16000
    return {"array": wav, "sampling_rate": int(sample_rate)}


def target_text(sample: dict) -> str:
    return str(sample.get("text") or sample.get("label") or "")


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.metadata).expanduser().resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    device = 0 if args.device == "cuda" and torch.cuda.is_available() else -1
    dtype = torch.float16 if device >= 0 else torch.float32
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        torch_dtype=dtype,
    )

    results = []
    for sample in metadata.get("samples", []):
        wav_path = Path(sample["adapter_wav"] if args.adapter_only else sample.get("baseline_wav", ""))
        target = target_text(sample)
        if not wav_path.exists():
            results.append(
                {
                    "audio": str(wav_path),
                    "target": target,
                    "transcript": "",
                    "correct_words": [],
                    "correct_count": 0,
                    "passed": False,
                    "error": "missing_audio",
                }
            )
            continue

        target_words = set(words(target))
        prediction = recognizer(read_audio(wav_path))
        transcript = str(prediction.get("text", "")).strip()
        transcript_words = set(words(transcript))
        correct = sorted(target_words & transcript_words)
        results.append(
            {
                "audio": str(wav_path),
                "target": target,
                "transcript": transcript,
                "target_words": sorted(target_words),
                "transcript_words": sorted(transcript_words),
                "correct_words": correct,
                "correct_count": len(correct),
                "passed": len(correct) >= args.min_correct_words,
            }
        )

    passing_samples = sum(1 for result in results if result["passed"])
    report = {
        "metadata": str(metadata_path),
        "model": args.model,
        "min_correct_words": args.min_correct_words,
        "min_passing_samples": args.min_passing_samples,
        "passing_samples": passing_samples,
        "passed": passing_samples >= args.min_passing_samples,
        "results": results,
    }

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
