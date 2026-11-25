"""Tail-local calibration from a paired anchor/TDOM memory.

Each projected-image tail remains an independent observation.  It can read the
two scene memories at the same ground pixel, but never attends to other tails.
This makes the module a lightweight evidence calibration step rather than a
second view-set fusion mechanism.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DualMemoryTailCalibrator(nn.Module):
    """Update each tail with the paired anchor/TDOM memory at one pixel."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        hidden_dim = max(int(embed_dim * ffn_ratio), 1)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        # The base model remains unchanged at initialisation; alpha gets a
        # direct gradient and opens the calibrated-tail path during training.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self,
        tails: torch.Tensor,
        tail_valid: torch.Tensor,
        anchor_memory: torch.Tensor,
        anchor_valid: torch.Tensor,
        tdom_memory: torch.Tensor,
        tdom_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return calibrated ``[B,Q,C,H,W]`` tails, alpha and delta ratio."""
        b, q, c, h, w = tails.shape
        if tail_valid.shape != (b, q, h, w):
            raise ValueError("tail_valid must have shape [B,Q,H,W]")

        tail_tokens = tails.permute(0, 3, 4, 1, 2).reshape(b * h * w, q, c)
        tail_valid_tokens = tail_valid.permute(0, 2, 3, 1).reshape(b * h * w, q)
        anchor_tokens = anchor_memory.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        tdom_tokens = tdom_memory.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        memory_tokens = torch.cat((anchor_tokens, tdom_tokens), dim=1)
        memory_valid = torch.stack((
            anchor_valid.reshape(b * h * w),
            tdom_valid.reshape(b * h * w),
        ), dim=1)

        # MultiheadAttention cannot accept rows with every key masked.  Such a
        # row is zeroed after attention; ordinarily TDOM makes it impossible.
        padding_mask = ~memory_valid
        all_invalid = padding_mask.all(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False
            memory_tokens = memory_tokens.clone()
            memory_tokens[all_invalid] = 0.0

        attended, _ = self.attn(
            self.norm_q(tail_tokens),
            self.norm_kv(memory_tokens),
            self.norm_kv(memory_tokens),
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        candidate = tail_tokens + attended
        delta = attended + self.ffn(self.norm_ffn(candidate))
        delta = delta.masked_fill(~tail_valid_tokens.unsqueeze(-1), 0.0)
        if all_invalid.any():
            delta[all_invalid] = 0.0

        alpha = self.alpha.to(device=tails.device, dtype=tails.dtype)
        output = tail_tokens + alpha * delta
        output = output.masked_fill(~tail_valid_tokens.unsqueeze(-1), 0.0)
        ratio = delta.float().norm(dim=-1).mean() / (
            tail_tokens.float().norm(dim=-1).mean().clamp_min(1e-6)
        )
        output = output.reshape(b, h, w, q, c).permute(0, 3, 4, 1, 2)
        return output, self.alpha.detach(), ratio.detach()
