"""Lightweight Swin-style spatial refinement for an anchor memory map.

This is intentionally a *refiner*, not another backbone: it preserves the
input resolution and channel count and alternates window and shifted-window
self-attention.  The final projection is zero initialised so a new branch is
an exact identity at initialisation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B,H,W,C] -> [B*nW, window_size*window_size, C]."""
    b, h, w, c = x.shape
    return (
        x.view(b, h // window_size, window_size, w // window_size, window_size, c)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size * window_size, c)
    )


def _window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    """[B*nW,N,C] -> [B,H,W,C]."""
    b = windows.shape[0] // ((height // window_size) * (width // window_size))
    c = windows.shape[-1]
    return (
        windows.view(b, height // window_size, width // window_size,
                     window_size, window_size, c)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(b, height, width, c)
    )


class _WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Swin-R embed_dim must be divisible by num_heads")
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        num_tokens = window_size * window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))
        coords_flatten = coords.flatten(1)
        relative = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        relative_index = relative.sum(-1)
        self.register_buffer("relative_position_index", relative_index, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(
            b_windows, n, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        scores = q @ k.transpose(-2, -1)
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(n, n, -1).permute(2, 0, 1).contiguous()
        scores = scores + relative_bias.unsqueeze(0)

        if attention_mask is not None:
            n_windows = attention_mask.shape[0]
            scores = scores.view(
                -1, n_windows, self.num_heads, n, n
            ) + attention_mask.unsqueeze(0).unsqueeze(2)
            scores = scores.view(-1, self.num_heads, n, n)

        if valid is not None:
            # Invalid padded/empty pixels cannot be keys or values.  Keep a
            # harmless zero key for an entirely empty window, then clear all
            # invalid query outputs below to avoid NaNs.
            valid = valid.bool()
            key_mask = ~valid[:, None, None, :]
            scores = scores.masked_fill(key_mask, -float("inf"))
            empty = ~valid.any(dim=-1)
            if empty.any():
                scores = scores.clone()
                scores[empty, :, :, 0] = 0.0
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        out = self.proj_drop(self.proj(out))
        if valid is not None:
            out = out.masked_fill(~valid.unsqueeze(-1), 0.0)
        return out


class _SwinRBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        if not 0 <= shift_size < window_size:
            raise ValueError("Swin-R shift_size must be in [0, window_size)")
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _shift_mask(height: int, width: int, window_size: int, shift_size: int,
                    device: torch.device) -> torch.Tensor:
        mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        value = 0
        for hs in h_slices:
            for ws in w_slices:
                mask[:, hs, ws, :] = value
                value += 1
        mask_windows = _window_partition(mask, window_size).squeeze(-1)
        diff = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return diff.masked_fill(diff != 0, -100.0).masked_fill(diff == 0, 0.0)

    def forward(self, x: torch.Tensor, valid_map: Optional[torch.Tensor]) -> torch.Tensor:
        # x is [B,H,W,C], valid_map is [B,H,W].
        b, h, w, c = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        x_pad = F.pad(x_norm, (0, 0, 0, pad_w, 0, pad_h))
        v_pad = None
        if valid_map is not None:
            v_pad = F.pad(valid_map, (0, pad_w, 0, pad_h), value=False)
        hp, wp = h + pad_h, w + pad_w
        if self.shift_size:
            shifted = torch.roll(x_pad, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_valid = torch.roll(v_pad, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if v_pad is not None else None
            attn_mask = self._shift_mask(hp, wp, self.window_size, self.shift_size, x.device)
        else:
            shifted, shifted_valid, attn_mask = x_pad, v_pad, None
        windows = _window_partition(shifted, self.window_size)
        valid_windows = (
            _window_partition(shifted_valid.unsqueeze(-1).to(dtype=x.dtype), self.window_size)
            .squeeze(-1).bool()
            if shifted_valid is not None else None
        )
        attended = self.attn(windows, valid_windows, attn_mask)
        shifted = _window_reverse(attended, self.window_size, hp, wp)
        if self.shift_size:
            shifted = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + shifted[:, :h, :w, :]
        x = x + self.mlp(self.norm2(x))
        if valid_map is not None:
            x = x.masked_fill(~valid_map.unsqueeze(-1), 0.0)
        return x


class SwinRRefiner(nn.Module):
    """Resolution-preserving shifted-window refinement of ``R0``.

    Returns ``R = R0 + alpha * delta`` and a detached relative-delta scalar
    useful for logging/auditing.  ``alpha`` is one learnable scalar per FPN
    scale because each scale owns an independent refiner in the parent layer.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        window_size: int = 8,
        num_heads: int = 8,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        alpha_init: float = 0.01,
    ):
        super().__init__()
        if window_size < 1:
            raise ValueError("Swin-R window_size must be >= 1")
        if depth < 1:
            raise ValueError("Swin-R depth must be >= 1")
        self.blocks = nn.ModuleList([
            _SwinRBlock(
                embed_dim, num_heads, window_size,
                0 if i % 2 == 0 else window_size // 2,
                mlp_ratio, dropout,
            )
            for i in range(depth)
        ])
        self.delta_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self, r0: torch.Tensor, valid_map: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if r0.ndim != 4:
            raise ValueError(f"Swin-R expects [B,C,H,W], got {tuple(r0.shape)}")
        x = r0.permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            x = block(x, valid_map)
        delta = self.delta_proj((x - r0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        refined = r0 + self.alpha.to(dtype=r0.dtype, device=r0.device) * delta
        if valid_map is not None:
            refined = refined.masked_fill(~valid_map.unsqueeze(1), 0.0)
            delta = delta.masked_fill(~valid_map.unsqueeze(1), 0.0)
        ratio = delta.float().norm() / r0.float().norm().clamp_min(1e-6)
        return refined, self.alpha.detach(), ratio.detach()
