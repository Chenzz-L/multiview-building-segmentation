"""Single scene memory from spatially contextualized unordered anchor views."""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.swin_r import SwinRRefiner


def _safe_padding(valid: torch.Tensor):
    padding = ~valid
    empty = ~valid.any(dim=1)
    if empty.any():
        padding = padding.clone()
        padding[empty, 0] = False
    return padding, empty


class _ViewSetBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        padding, empty = _safe_padding(valid)
        normed = self.norm1(tokens)
        update, _ = self.attn(normed, normed, normed, key_padding_mask=padding, need_weights=False)
        tokens = tokens + update
        tokens = tokens + self.ffn(self.norm2(tokens))
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        if empty.any():
            tokens[empty] = 0.0
        return tokens


class SingleRSwinSet(nn.Module):
    """Shared per-view Swin -> view SAB -> one-query PMA -> residual R."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        spatial_window_size: int = 8,
        spatial_depth: int = 2,
        view_depth: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        spatial_alpha_init: float = 0.01,
        beta_init: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.spatial = SwinRRefiner(
            embed_dim, spatial_window_size, num_heads, spatial_depth,
            mlp_ratio, dropout, spatial_alpha_init,
        )
        self.view_blocks = nn.ModuleList([
            _ViewSetBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(view_depth)
        ])
        self.pool_seed = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pool_seed, std=0.02)
        self.pool_q_norm = nn.LayerNorm(embed_dim)
        self.pool_kv_norm = nn.LayerNorm(embed_dim)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pool_ffn_norm = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.pool_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    @staticmethod
    def _masked_mean(anchors: torch.Tensor, valid: torch.Tensor):
        weights = valid.unsqueeze(2).to(anchors.dtype)
        count = weights.sum(dim=1)
        mean = (anchors * weights).sum(dim=1) / count.clamp_min(1.0)
        mean = torch.where(count > 0, mean, torch.zeros_like(mean))
        return mean, count

    def forward(self, anchors: torch.Tensor, valid: torch.Tensor):
        b, k, c, h, w = anchors.shape
        if c != self.embed_dim or valid.shape != (b, k, h, w):
            raise ValueError("Invalid anchor/valid shapes for SingleRSwinSet")
        r0, count = self._masked_mean(anchors, valid)

        flat = anchors.reshape(b * k, c, h, w)
        flat_valid = valid.reshape(b * k, h, w)
        spatial, spatial_alpha, spatial_ratio = self.spatial(flat, flat_valid)
        spatial = spatial.reshape(b, k, c, h, w)

        tokens = spatial.permute(0, 3, 4, 1, 2).reshape(b * h * w, k, c)
        token_valid = valid.permute(0, 2, 3, 1).reshape(b * h * w, k)
        for block in self.view_blocks:
            tokens = block(tokens, token_valid)

        padding, empty = _safe_padding(token_valid)
        query = self.pool_seed.expand(tokens.shape[0], -1, -1)
        pooled, weights = self.pool_attn(
            self.pool_q_norm(query), self.pool_kv_norm(tokens), self.pool_kv_norm(tokens),
            key_padding_mask=padding, need_weights=True, average_attn_weights=True,
        )
        pooled = pooled + self.pool_ffn(self.pool_ffn_norm(pooled))
        pooled = pooled[:, 0]
        pooled = pooled.masked_fill(empty.unsqueeze(-1), 0.0)
        rhat = pooled.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        memory = r0 + self.beta.to(r0.dtype) * (rhat - r0)
        valid_map = count[:, 0] > 0
        memory = memory.masked_fill(~valid_map.unsqueeze(1), 0.0)
        contextualized = tokens.reshape(b, h, w, k, c).permute(0, 3, 4, 1, 2)
        contextualized = contextualized.masked_fill(~valid.unsqueeze(2), 0.0)
        pool_mean = weights[:, 0].reshape(b, h, w, k).mean(dim=(0, 1, 2)).detach()
        return (
            memory, count / float(max(k, 1)), contextualized, valid,
            spatial_alpha, spatial_ratio, pool_mean,
        )
