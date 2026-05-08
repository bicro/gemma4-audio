#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from gemma_kyutai_tts import GemmaMimiHeadConfig, GemmaToMimiCodecHead, count_trainable_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gemma-hidden to Kyutai Mimi codec-token head.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--valid_cache", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--save_every_steps", type=int, default=250)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--adapter_hidden_size", type=int, default=1536)
    parser.add_argument("--bridge_num_heads", type=int, default=8)
    parser.add_argument("--num_bridge_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max_audio_frames", type=int, default=128)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--mimi_cardinality", type=int, default=2048)
    parser.add_argument("--codebook_weights", default="2.0,1.5,1.0,1.0,0.75,0.75,0.5,0.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--log_every_steps", type=int, default=10)
    return parser.parse_args()


class CachedGemmaMimiDataset(Dataset):
    def __init__(self, cache_path: str) -> None:
        self.cache_path = Path(cache_path).expanduser().resolve()
        try:
            payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(self.cache_path, map_location="cpu")
        self.payload = payload
        self.items = payload["items"]
        if not self.items:
            raise ValueError(f"empty cache: {self.cache_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def collate(batch: list[dict]) -> dict:
    layer_count = len(batch[0]["hidden_states"])
    hidden_size = batch[0]["hidden_states"][-1].shape[-1]
    max_seq = max(item["hidden_states"][-1].shape[0] for item in batch)
    max_frames = max(item["mimi_codes"].shape[-1] for item in batch)
    codebooks = batch[0]["mimi_codes"].shape[0]

    hidden_layers = []
    for layer_index in range(layer_count):
        layer = torch.zeros((len(batch), max_seq, hidden_size), dtype=torch.bfloat16)
        for batch_index, item in enumerate(batch):
            value = item["hidden_states"][layer_index]
            layer[batch_index, : value.shape[0], :] = value
        hidden_layers.append(layer)

    targets = torch.full((len(batch), codebooks, max_frames), -100, dtype=torch.long)
    frame_lengths = []
    for batch_index, item in enumerate(batch):
        codes = item["mimi_codes"]
        frame_lengths.append(int(codes.shape[-1]))
        targets[batch_index, : codes.shape[0], : codes.shape[-1]] = codes

    return {
        "hidden_states": tuple(hidden_layers),
        "targets": targets,
        "frame_lengths": torch.tensor(frame_lengths, dtype=torch.long),
        "texts": [item["text"] for item in batch],
    }


def parse_weights(weights: str, codebooks: int, device: torch.device) -> torch.Tensor:
    values = [float(item) for item in weights.split(",") if item.strip()]
    if not values:
        values = [1.0]
    if len(values) < codebooks:
        values.extend([values[-1]] * (codebooks - len(values)))
    return torch.tensor(values[:codebooks], device=device, dtype=torch.float32)


def codec_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, dict]:
    losses = []
    stats = {}
    vocab = logits.shape[-1]
    for codebook in range(targets.shape[1]):
        loss = F.cross_entropy(
            logits[:, codebook, :, :].reshape(-1, vocab).float(),
            targets[:, codebook, :].reshape(-1),
            ignore_index=-100,
        )
        losses.append(loss * weights[codebook])
        stats[f"ce_k{codebook}"] = float(loss.detach().cpu())
    return torch.stack(losses).sum() / weights[: targets.shape[1]].sum(), stats


def save_checkpoint(model: GemmaToMimiCodecHead, config: GemmaMimiHeadConfig, output_dir: Path, step: int, meta: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"mimi-head-step-{step}.pt"
    torch.save(
        {
            "head_state_dict": model.state_dict(),
            "head_config": asdict(config),
            "global_step": step,
            "meta": meta,
        },
        path,
    )
    (output_dir / "head_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CachedGemmaMimiDataset(args.train_cache)
    valid_dataset = CachedGemmaMimiDataset(args.valid_cache) if args.valid_cache else None
    gemma_hidden_size = int(train_dataset.payload["gemma_hidden_size"])
    gemma_layer_count = int(train_dataset.payload["gemma_layer_count"])

    config = GemmaMimiHeadConfig(
        gemma_hidden_size=gemma_hidden_size,
        adapter_hidden_size=args.adapter_hidden_size,
        gemma_layer_count=gemma_layer_count,
        bridge_num_heads=args.bridge_num_heads,
        num_bridge_layers=args.num_bridge_layers,
        dropout=args.dropout,
        max_audio_frames=args.max_audio_frames,
        mimi_codebooks=args.num_codebooks,
        mimi_cardinality=args.mimi_cardinality,
    )
    head = GemmaToMimiCodecHead(config).to(device)
    global_step = 0
    if args.resume_checkpoint:
        try:
            checkpoint = torch.load(
                Path(args.resume_checkpoint).expanduser().resolve(),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(Path(args.resume_checkpoint).expanduser().resolve(), map_location="cpu")
        head.load_state_dict(checkpoint["head_state_dict"])
        global_step = int(checkpoint.get("global_step", 0))
        print(f"resumed {args.resume_checkpoint} at global_step={global_step}")

    trainable, total = count_trainable_parameters(head)
    print(f"Gemma is cached/frozen; Mimi is frozen decoder only; optimizer has head params only.")
    print(f"Head parameters: {trainable:,} trainable / {total:,} total")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    valid_loader = (
        DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        if valid_dataset is not None
        else None
    )

    optimizer = AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = parse_weights(args.codebook_weights, args.num_codebooks, device)
    head.train()
    optimizer.zero_grad(set_to_none=True)

    meta = {
        "train_cache": str(Path(args.train_cache).expanduser().resolve()),
        "valid_cache": str(Path(args.valid_cache).expanduser().resolve()) if args.valid_cache else None,
        "loss": "weighted per-codebook cross entropy over Mimi tokens",
    }

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_loader):
            hidden_states = tuple(layer.to(device=device) for layer in batch["hidden_states"])
            targets = batch["targets"].to(device=device)
            logits = head(hidden_states, target_length=targets.shape[-1])
            loss, stats = codec_cross_entropy(logits, targets, weights)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every_steps == 0:
                    stat_str = " ".join(f"{key}={value:.3f}" for key, value in list(stats.items())[:4])
                    print(
                        f"epoch={epoch} global_step={global_step} "
                        f"loss={(loss.item() * args.gradient_accumulation_steps):.4f} {stat_str}",
                        flush=True,
                    )

                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    ckpt = save_checkpoint(head, config, output_dir, global_step, meta)
                    print(f"saved {ckpt}", flush=True)

                if args.max_steps and global_step >= args.max_steps:
                    break

        ckpt = save_checkpoint(head, config, output_dir, global_step, meta)
        print(f"saved {ckpt}", flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break

        if valid_loader is not None:
            head.eval()
            valid_losses = []
            with torch.inference_mode():
                for batch in valid_loader:
                    hidden_states = tuple(layer.to(device=device) for layer in batch["hidden_states"])
                    targets = batch["targets"].to(device=device)
                    logits = head(hidden_states, target_length=targets.shape[-1])
                    valid_loss, _ = codec_cross_entropy(logits, targets, weights)
                    valid_losses.append(float(valid_loss.detach().cpu()))
            if valid_losses:
                print(f"valid_loss={sum(valid_losses) / len(valid_losses):.4f}", flush=True)
            head.train()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
