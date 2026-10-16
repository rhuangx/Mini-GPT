"""The baseline GPT: pre-norm RMSNorm, RoPE, SwiGLU, full causal MHA, no biases, tied embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# Targets equal to IGNORE_INDEX drop out of the loss: the val set's <|bos|>
# targets and padding.
IGNORE_INDEX = -100


@dataclass
class GPTConfig:
    vocab_size: int
    context: int
    d_model: int
    n_layers: int
    head_dim: int
    rope_base: float

    def __post_init__(self) -> None:
        assert self.d_model % self.head_dim == 0, "d_model must be a multiple of head_dim"
        assert self.head_dim % 2 == 0, "RoPE rotates pairs of dims"

    @property
    def n_heads(self) -> int:
        return self.d_model // self.head_dim

    @property
    def mlp_hidden(self) -> int:
        """8/3 * d_model, rounded to a multiple of 32."""
        return 32 * round(8 * self.d_model / 3 / 32)

    def param_counts(self) -> dict[str, int]:
        """Parameter counts as GPT builds them; the tied LM head is the embedding."""
        d = self.d_model
        per_layer = 4 * d * d + 3 * d * self.mlp_hidden + 2 * d  # attention, SwiGLU, two norm gains
        return {"embedding": self.vocab_size * d, "non_embedding": self.n_layers * per_layer + d}


def rope_tables(cfg: GPTConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin, each [context, head_dim / 2]; pair i turns at rope_base^(-2i / head_dim)."""
    inv_freq = cfg.rope_base ** (-torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim)
    angles = torch.outer(torch.arange(cfg.context).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x [B, H, T, d] by position, pairing dim i with dim i + d/2."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        # [B, T, 3D] -> 3 x [B, H, T, head_dim]
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # GPT-2 init: N(0, 0.02), with the two projections that write to the
        # residual stream scaled down by sqrt(2 * n_layers).
        for p in self.parameters():
            if p.ndim == 2:
                nn.init.normal_(p, std=0.02)
        for block in self.blocks:
            for p in (block.attn.proj.weight, block.mlp.down.weight):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-token NLL in nats, [B, T]; 0 where the target is IGNORE_INDEX."""
        t = idx.size(1)
        cos, sin = self.cos[:t], self.sin[:t]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = F.linear(self.norm(x), self.embed.weight)  # the tied LM head
        nll = F.cross_entropy(
            logits.flatten(0, 1).float(), targets.flatten(), ignore_index=IGNORE_INDEX, reduction="none"
        )
        return nll.view_as(targets)
