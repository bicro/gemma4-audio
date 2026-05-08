from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GemmaMimiHeadConfig:
    gemma_hidden_size: int = 2560
    adapter_hidden_size: int = 1536
    gemma_layer_count: int = 6
    bridge_num_heads: int = 8
    num_bridge_layers: int = 4
    dropout: float = 0.05
    max_audio_frames: int = 256
    mimi_codebooks: int = 8
    mimi_cardinality: int = 2048


class GemmaToMimiCodecHead(nn.Module):
    """Map frozen Gemma states to Mimi codec-token logits.

    This is not a TTS pipeline wrapper. It predicts Mimi discrete audio tokens
    directly from Gemma hidden states; a frozen Mimi decoder turns those tokens
    into waveform audio.
    """

    def __init__(self, config: GemmaMimiHeadConfig) -> None:
        super().__init__()
        self.config = config
        if config.adapter_hidden_size % config.bridge_num_heads != 0:
            raise ValueError("adapter_hidden_size must be divisible by bridge_num_heads")

        self.layer_weights = nn.Parameter(torch.zeros(config.gemma_layer_count))
        self.input_norm = nn.LayerNorm(config.gemma_hidden_size)
        self.input_projection = nn.Linear(config.gemma_hidden_size, config.adapter_hidden_size)
        self.audio_queries = nn.Parameter(
            torch.randn(config.max_audio_frames, config.adapter_hidden_size) / math.sqrt(config.adapter_hidden_size)
        )
        self.query_norm = nn.LayerNorm(config.adapter_hidden_size)
        self.key_value_norm = nn.LayerNorm(config.adapter_hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.adapter_hidden_size,
            num_heads=config.bridge_num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.adapter_hidden_size,
            nhead=config.bridge_num_heads,
            dim_feedforward=config.adapter_hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.bridge = nn.TransformerEncoder(encoder_layer, num_layers=config.num_bridge_layers)
        self.output_norm = nn.LayerNorm(config.adapter_hidden_size)
        self.codebook_heads = nn.ModuleList(
            [nn.Linear(config.adapter_hidden_size, config.mimi_cardinality) for _ in range(config.mimi_codebooks)]
        )

    def _mix_layers(self, gemma_hidden_states) -> torch.Tensor:
        if isinstance(gemma_hidden_states, torch.Tensor):
            return gemma_hidden_states
        if not isinstance(gemma_hidden_states, (tuple, list)):
            raise TypeError("gemma_hidden_states must be a tensor or hidden-state tuple/list")
        if len(gemma_hidden_states) < self.config.gemma_layer_count:
            raise ValueError(
                f"need at least {self.config.gemma_layer_count} hidden states, got {len(gemma_hidden_states)}"
            )
        selected = gemma_hidden_states[-self.config.gemma_layer_count :]
        dtype = self.input_projection.weight.dtype
        device = self.input_projection.weight.device
        weights = torch.softmax(self.layer_weights, dim=0).to(device=device, dtype=dtype)
        mixed = selected[-1].to(device=device, dtype=dtype) * weights[-1]
        for hidden_state, weight in zip(selected[:-1], weights[:-1]):
            mixed = mixed + hidden_state.to(device=device, dtype=dtype) * weight
        return mixed

    @staticmethod
    def _resize_sequence(hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
        if hidden_states.shape[1] == target_length:
            return hidden_states
        return torch.nn.functional.interpolate(
            hidden_states.transpose(1, 2).float(),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2).to(dtype=hidden_states.dtype)

    def forward(self, gemma_hidden_states, target_length: int) -> torch.Tensor:
        if target_length > self.config.max_audio_frames:
            raise ValueError(
                f"target_length={target_length} exceeds max_audio_frames={self.config.max_audio_frames}"
            )

        mixed = self._mix_layers(gemma_hidden_states)
        if mixed.ndim != 3:
            raise ValueError("gemma hidden states must be shaped [batch, seq, hidden]")
        if mixed.shape[-1] != self.config.gemma_hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.gemma_hidden_size}, got {mixed.shape[-1]}"
            )

        keys_values = self.input_projection(self.input_norm(mixed))
        batch_size = keys_values.shape[0]
        learned_queries = self.audio_queries[:target_length].unsqueeze(0).expand(batch_size, -1, -1)
        semantic_queries = self._resize_sequence(keys_values, target_length)
        queries = learned_queries.to(dtype=keys_values.dtype) + semantic_queries

        attended, _ = self.cross_attention(
            query=self.query_norm(queries),
            key=self.key_value_norm(keys_values),
            value=keys_values,
            need_weights=False,
        )
        hidden = self.bridge(queries + attended)
        hidden = self.output_norm(hidden)
        logits = torch.stack([head(hidden) for head in self.codebook_heads], dim=1)
        return logits


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in module.parameters():
        n = parameter.numel()
        total += n
        if parameter.requires_grad:
            trainable += n
    return trainable, total

