#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create audio-only prompt-contained manifests by prepending a spoken instruction WAV."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix_audio", required=True)
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--gap_sec", type=float, default=0.25)
    parser.add_argument("--sample_rate", type=int, default=16000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_mono_resampled(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        target_len = max(1, int(round(audio.shape[0] * sample_rate / source_rate)))
        source_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        audio = np.interp(target_x, source_x, audio).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    prefix = read_mono_resampled(Path(args.prefix_audio).expanduser().resolve(), args.sample_rate)
    gap = np.zeros(int(round(args.gap_sec * args.sample_rate)), dtype=np.float32)
    audio_root = output_dir / "audio_prompt_contained"
    audio_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        rows = read_jsonl(input_dir / f"{split}_with_mimi_codes.jsonl")
        rewritten = []
        split_audio_dir = audio_root / split
        split_audio_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows):
            source_audio = Path(str(row["audio"])).expanduser().resolve()
            target_audio = read_mono_resampled(source_audio, args.sample_rate)
            prompt_audio = np.concatenate([prefix, gap, target_audio])
            segment_id = row.get("segment_id") or f"{split}_{index:06d}"
            output_audio = split_audio_dir / f"{segment_id}_audio_prompt.wav"
            sf.write(output_audio, prompt_audio, args.sample_rate)
            next_row = dict(row)
            next_row["source_audio"] = str(source_audio)
            next_row["audio"] = str(output_audio)
            next_row["audio_prompt_prefix_sec"] = float(len(prefix) / args.sample_rate)
            next_row["audio_prompt_gap_sec"] = float(len(gap) / args.sample_rate)
            next_row["audio_prompt_duration_sec"] = float(len(prompt_audio) / args.sample_rate)
            rewritten.append(next_row)
        write_jsonl(output_dir / f"{split}_with_mimi_codes.jsonl", rewritten)
        summary[split] = len(rewritten)

    (output_dir / "audio_prompt_manifest_summary.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "prefix_audio": str(Path(args.prefix_audio).expanduser().resolve()),
                "sample_rate": args.sample_rate,
                "gap_sec": args.gap_sec,
                "splits": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
