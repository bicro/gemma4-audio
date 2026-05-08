#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForMultimodalLM, AutoProcessor
except ImportError:  # transformers builds before the Auto class rename
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration as AutoModelForMultimodalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen Gemma hidden states for Mimi-code training.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gemma_model_path", default="google/gemma-4-E4B-it")
    parser.add_argument("--gemma_layer_count", type=int, default=6)
    parser.add_argument("--gemma_prompt_template", default="Say this naturally as speech:\n{text}")
    parser.add_argument(
        "--input_mode",
        choices=("text", "audio", "both"),
        default="text",
        help="Cache text-conditioned states, audio-conditioned states, or both for the same Mimi targets.",
    )
    parser.add_argument(
        "--audio_prompt_template",
        default="Listen to the attached audio and repeat the spoken sentence naturally as speech.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


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


def cache_item(
    *,
    row: dict,
    index: int,
    condition: str,
    hidden_states,
    gemma_layer_count: int,
) -> tuple[dict, int]:
    selected = hidden_states[-gemma_layer_count:]
    cached_layers = [layer[0].detach().cpu().to(torch.bfloat16).contiguous() for layer in selected]
    hidden_size = int(cached_layers[-1].shape[-1])
    item = {
        "text": row["text"],
        "audio": row.get("audio"),
        "duration_sec": row.get("duration_sec"),
        "segment_id": f"{row.get('segment_id', f'row_{index:06d}')}_{condition}",
        "condition": condition,
        "mimi_codes": torch.tensor(row["mimi_codes"], dtype=torch.long),
        "mimi_code_shape": row.get("mimi_code_shape"),
        "hidden_states": cached_layers,
    }
    return item, hidden_size


def build_audio_message(audio_path: Path, instruction: str) -> list[dict]:
    content = []
    if instruction.strip():
        content.append({"type": "text", "text": instruction})
    content.append({"type": "audio", "audio": str(audio_path)})
    return [{"role": "user", "content": content}]


def cache_text_only(args, rows: list[dict], device: torch.device, dtype: torch.dtype) -> tuple[list[dict], int | None]:
    tokenizer = AutoTokenizer.from_pretrained(args.gemma_model_path, extra_special_tokens={})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gemma = AutoModelForCausalLM.from_pretrained(
        args.gemma_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    gemma.eval()
    for parameter in gemma.parameters():
        parameter.requires_grad_(False)

    items = []
    hidden_size = None
    for index, row in enumerate(rows):
        prompt = args.gemma_prompt_template.format(text=row["text"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            outputs = gemma(**inputs, output_hidden_states=True, use_cache=False)
        item, hidden_size = cache_item(
            row=row,
            index=index,
            condition="text",
            hidden_states=outputs.hidden_states,
            gemma_layer_count=args.gemma_layer_count,
        )
        items.append(item)
        if index % 25 == 0:
            print(f"cached {index + 1}/{len(rows)} text")
    return items, hidden_size


def cache_multimodal(args, rows: list[dict], device: torch.device, dtype: torch.dtype) -> tuple[list[dict], int | None]:
    processor = AutoProcessor.from_pretrained(args.gemma_model_path, padding_side="left")
    gemma = AutoModelForMultimodalLM.from_pretrained(
        args.gemma_model_path,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    gemma.eval()
    for parameter in gemma.parameters():
        parameter.requires_grad_(False)

    items = []
    hidden_size = None
    for index, row in enumerate(rows):
        if args.input_mode in {"text", "both"}:
            prompt = args.gemma_prompt_template.format(text=row["text"])
            inputs = processor(text=prompt, return_tensors="pt", padding=True)
            inputs = move_inputs_to_device(inputs, device=device, dtype=dtype)
            with torch.inference_mode():
                outputs = gemma(**inputs, output_hidden_states=True, use_cache=False)
            item, hidden_size = cache_item(
                row=row,
                index=index,
                condition="text",
                hidden_states=outputs.hidden_states,
                gemma_layer_count=args.gemma_layer_count,
            )
            items.append(item)

        if args.input_mode in {"audio", "both"}:
            audio_path = Path(str(row["audio"])).expanduser().resolve()
            audio, _sample_rate = read_audio_16k(audio_path)
            messages = build_audio_message(audio_path, args.audio_prompt_template.format(text=row["text"]))
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, audio=audio[None, :], return_tensors="pt", padding=True)
            inputs = move_inputs_to_device(inputs, device=device, dtype=dtype)
            with torch.inference_mode():
                outputs = gemma(**inputs, output_hidden_states=True, use_cache=False)
            item, hidden_size = cache_item(
                row=row,
                index=index,
                condition="audio",
                hidden_states=outputs.hidden_states,
                gemma_layer_count=args.gemma_layer_count,
            )
            items.append(item)

        if index % 25 == 0:
            print(f"cached {index + 1}/{len(rows)} mode={args.input_mode} items={len(items)}")
    return items, hidden_size


def main() -> int:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)

    rows = read_jsonl(input_jsonl, args.limit)
    if args.input_mode == "text":
        items, hidden_size = cache_text_only(args, rows, device, dtype)
    else:
        items, hidden_size = cache_multimodal(args, rows, device, dtype)

    payload = {
        "gemma_model_path": args.gemma_model_path,
        "gemma_layer_count": args.gemma_layer_count,
        "gemma_prompt_template": args.gemma_prompt_template,
        "audio_prompt_template": args.audio_prompt_template,
        "input_mode": args.input_mode,
        "gemma_hidden_size": hidden_size,
        "source_jsonl": str(input_jsonl),
        "items": items,
    }
    torch.save(payload, output)
    print(json.dumps({"output": str(output), "items": len(items), "hidden_size": hidden_size}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
