#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode teacher WAVs into Kyutai Mimi codec tokens.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--input_suffix", default="_raw.jsonl")
    parser.add_argument("--output_suffix", default="_with_mimi_codes.jsonl")
    parser.add_argument("--audio_key", default="audio")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--limit_per_split", type=int, default=0)
    parser.add_argument("--mimi_repo", default=loaders.DEFAULT_REPO)
    parser.add_argument("--mimi_weight", default=loaders.MIMI_NAME)
    parser.add_argument("--write_recon_samples", type=int, default=2)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    wav, _ = librosa.load(path, sr=sample_rate, mono=True)
    wav_tensor = torch.from_numpy(wav).float().view(1, 1, -1)
    return wav_tensor


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    mimi_weight = hf_hub_download(args.mimi_repo, args.mimi_weight)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(args.num_codebooks)
    mimi.eval()

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mimi_repo": args.mimi_repo,
        "mimi_weight": args.mimi_weight,
        "sample_rate": loaders.SAMPLE_RATE,
        "frame_rate": loaders.FRAME_RATE,
        "num_codebooks": args.num_codebooks,
        "splits": {},
    }

    recon_dir = output_dir / "mimi_recon_samples"
    for split in args.splits:
        rows = read_jsonl(input_dir / f"{split}{args.input_suffix}", args.limit_per_split)
        encoded_rows: list[dict] = []
        for index, row in enumerate(rows):
            audio_path = Path(row[args.audio_key]).expanduser()
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            wav = load_audio(audio_path, loaders.SAMPLE_RATE).to(device)
            with torch.inference_mode():
                codes = mimi.encode(wav)[0].detach().cpu().long()
            encoded = dict(row)
            encoded["mimi_codes"] = codes.tolist()
            encoded["mimi_code_shape"] = list(codes.shape)
            encoded["mimi_num_codebooks"] = int(args.num_codebooks)
            encoded["mimi_frame_rate"] = float(loaders.FRAME_RATE)
            encoded_rows.append(encoded)

            if index < args.write_recon_samples:
                with torch.inference_mode():
                    recon = mimi.decode(codes.unsqueeze(0).to(device))[0, 0].detach().cpu().float().numpy()
                recon_dir.mkdir(parents=True, exist_ok=True)
                sf.write(recon_dir / f"{split}_{index:03d}_mimi_recon.wav", recon, loaders.SAMPLE_RATE)

        write_jsonl(output_dir / f"{split}{args.output_suffix}", encoded_rows)
        summary["splits"][split] = {
            "rows": len(encoded_rows),
            "frames_min": min((len(row["mimi_codes"][0]) for row in encoded_rows), default=0),
            "frames_max": max((len(row["mimi_codes"][0]) for row in encoded_rows), default=0),
        }
        print(json.dumps({split: summary["splits"][split]}, sort_keys=True))

    (output_dir / "mimi_prepare_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

