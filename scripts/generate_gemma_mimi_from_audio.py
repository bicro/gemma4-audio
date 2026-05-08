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
from transformers import AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:  # transformers builds before the Auto class rename
    from transformers import Gemma4ForConditionalGeneration as AutoModelForMultimodalLM

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from gemma_kyutai_tts import GemmaMimiHeadConfig, GemmaToMimiCodecHead  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WAVs from Gemma audio-input states -> Mimi codec head.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audio", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--gemma_model_path", default="")
    parser.add_argument(
        "--instruction",
        default="Listen to the attached audio and repeat the spoken sentence naturally as speech.",
    )
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


def read_audio_16k(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        target_len = max(1, int(round(audio.shape[0] * 16000 / sample_rate)))
        source_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        audio = np.interp(target_x, source_x, audio).astype(np.float32)
        sample_rate = 16000
    return audio.astype(np.float32, copy=False), sample_rate


def move_inputs_to_device(inputs, device: torch.device, dtype: torch.dtype):
    moved = {}
    float_keys = {"pixel_values", "pixel_values_videos", "input_features"}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if key in float_keys and torch.is_floating_point(value):
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved


def build_audio_inputs(processor, messages: list[dict], audio: np.ndarray):
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if "input_features" in inputs:
            return inputs
    except Exception as exc:
        print(f"apply_chat_template(tokenize=True) failed, falling back to processor(...): {exc}", file=sys.stderr)

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audio_input = audio.astype(np.float32, copy=False)
    return processor(
        text=prompt,
        audio=audio_input[None, :],
        return_tensors="pt",
        padding=True,
    )


def build_audio_message(audio_path: Path, instruction: str) -> list[dict]:
    content = []
    if instruction.strip():
        content.append({"type": "text", "text": instruction})
    content.append({"type": "audio", "audio": str(audio_path)})
    return [{"role": "user", "content": content}]


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

    processor = AutoProcessor.from_pretrained(gemma_model_path, padding_side="left")
    gemma = AutoModelForMultimodalLM.from_pretrained(
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
    labels = list(args.label)
    while len(labels) < len(args.audio):
        labels.append("")

    metadata = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "gemma_model_path": gemma_model_path,
        "instruction": args.instruction,
        "mimi_repo": args.mimi_repo,
        "mimi_weight": args.mimi_weight,
        "target_frames": target_frames,
        "frame_rate": loaders.FRAME_RATE,
        "input_mode": "audio_only_content_plus_instruction" if args.instruction.strip() else "audio_only_no_text_instruction",
        "samples": [],
    }

    for index, (audio_arg, label) in enumerate(zip(args.audio, labels)):
        audio_path = Path(audio_arg).expanduser().resolve()
        audio, sample_rate = read_audio_16k(audio_path)
        messages = build_audio_message(audio_path, args.instruction)
        inputs = build_audio_inputs(processor, messages, audio)
        inputs = move_inputs_to_device(inputs, device=device, dtype=dtype)
        with torch.inference_mode():
            outputs = gemma(**inputs, output_hidden_states=True, use_cache=False)
            logits = head(outputs.hidden_states[-head.config.gemma_layer_count :], target_length=target_frames)
            codes = sample_codes(logits, args.temperature).long()
            wav = mimi.decode(codes)[0, 0].detach().cpu().float().numpy()

        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        adapter_path = output_dir / f"sample_{index:02d}_gemma_audio_input_mimi_head.wav"
        input_copy_path = output_dir / f"sample_{index:02d}_qwen_tts_input.wav"
        write_wav(adapter_path, wav, loaders.SAMPLE_RATE)
        write_wav(input_copy_path, audio, sample_rate)

        metadata["samples"].append(
            {
                "label": label,
                "input_audio": str(input_copy_path),
                "source_audio": str(audio_path),
                "source_sample_rate": sample_rate,
                "adapter_wav": str(adapter_path),
                "sample_rate": loaders.SAMPLE_RATE,
                "adapter_audio_samples": int(wav.shape[0]),
                "adapter_code_shape": list(codes[0].shape),
                "rms": float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0,
                "peak": float(np.max(np.abs(wav))) if wav.size else 0.0,
                "input_rms": float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0,
                "input_peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
                "input_duration_sec": float(len(audio) / sample_rate) if sample_rate else 0.0,
            }
        )

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
