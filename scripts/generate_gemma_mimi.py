#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from gemma_kyutai_tts import GemmaMimiHeadConfig, GemmaToMimiCodecHead  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WAVs from Gemma -> Mimi codec head.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--gemma_model_path", default="")
    parser.add_argument("--gemma_prompt_template", default="Say this naturally as speech:\n{text}")
    parser.add_argument("--seconds", type=float, default=3.6)
    parser.add_argument("--target_frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--mimi_repo", default=loaders.DEFAULT_REPO)
    parser.add_argument("--mimi_weight", default=loaders.MIMI_NAME)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_head(path: Path, device: torch.device):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    config = GemmaMimiHeadConfig(**payload["head_config"])
    head = GemmaToMimiCodecHead(config)
    head.load_state_dict(payload["head_state_dict"])
    head.to(device)
    head.eval()
    return head, payload


def sample_codes(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    scores = logits.float() / max(float(temperature), 1e-5)
    probs = torch.softmax(scores, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).reshape(probs.shape[:-1])
    return sampled


def write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, sample_rate)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)

    head, payload = load_head(Path(args.checkpoint).expanduser().resolve(), device)
    gemma_model_path = args.gemma_model_path or payload.get("meta", {}).get("gemma_model_path") or "google/gemma-4-E4B-it"
    tokenizer = AutoTokenizer.from_pretrained(gemma_model_path, extra_special_tokens={})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gemma = AutoModelForCausalLM.from_pretrained(
        gemma_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    gemma.eval()
    for parameter in gemma.parameters():
        parameter.requires_grad_(False)

    mimi_weight = hf_hub_download(args.mimi_repo, args.mimi_weight)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(args.num_codebooks)
    mimi.eval()

    target_frames = args.target_frames or int(round(args.seconds * loaders.FRAME_RATE))
    metadata = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "gemma_model_path": gemma_model_path,
        "mimi_repo": args.mimi_repo,
        "mimi_weight": args.mimi_weight,
        "target_frames": target_frames,
        "frame_rate": loaders.FRAME_RATE,
        "samples": [],
    }

    for index, text in enumerate(args.text):
        prompt = args.gemma_prompt_template.format(text=text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            outputs = gemma(**inputs, output_hidden_states=True, use_cache=False)
            logits = head(outputs.hidden_states[-head.config.gemma_layer_count :], target_length=target_frames)
            codes = sample_codes(logits, args.temperature).long()
            wav = mimi.decode(codes)[0, 0].detach().cpu().float().numpy()

        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        adapter_path = output_dir / f"sample_{index:02d}_gemma_mimi_head.wav"
        write_wav(adapter_path, wav, loaders.SAMPLE_RATE)
        metadata["samples"].append(
            {
                "text": text,
                "adapter_wav": str(adapter_path),
                "sample_rate": loaders.SAMPLE_RATE,
                "adapter_audio_samples": int(wav.shape[0]),
                "adapter_code_shape": list(codes[0].shape),
                "rms": float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0,
                "peak": float(np.max(np.abs(wav))) if wav.size else 0.0,
            }
        )

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
