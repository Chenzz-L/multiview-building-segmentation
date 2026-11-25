"""R-conditioned local cross-attention for projected tail features.

For every ground-plane location, the compact anchor memory ``R(x, y)`` is the
query.  Keys and values are taken from a local neighbourhood in every tail
view.  The softmax is therefore over ``tail_view x relative_offset`` rather
than only over tail views at the same pixel.

The reduction is implemented as an online softmax.  Besides avoiding a large
``[B, Q, window**2, heads, H, W]`` tensor, this makes fixed-Q and chunked
all-tail inference mathematically equivalent (up to floating-point order).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RConditionedLocalTailAttention(nn.Module):
    """Let one compact R query read local candidates from all tail views."""

    def __init__(
        self,
        embed_dim: int = 256,
        attention_dim: int = 64,
        num_heads: int = 4,
        window_size: int = 3,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        if window_size < 3 or window_size % 2 != 1:
            raise ValueError("window_size must be an odd integer >= 3")
        if attention_dim < 1 or attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be positive and divisible by num_heads")
        if ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be > 0")

        self.embed_dim = int(embed_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.window_size = int(window_size)
        self.radius = self.window_size // 2
        self.eps = float(eps)

        self.query_norm = nn.LayerNorm(embed_dim)
        self.tail_norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, attention_dim)
        self.k_proj = nn.Linear(embed_dim, attention_dim)
        self.v_proj = nn.Linear(embed_dim, attention_dim)
        self.out_proj = nn.Linear(attention_dim, embed_dim)
        self.out_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ffn_norm = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        # No view-id embedding is used: medoid ordering is arbitrary.  This
        # bias only identifies the relative ground-plane offset.
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, self.window_size * self.window_size)
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def init_stream(
        self,
        query_map: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Initialise an exact online-softmax state for one feature scale."""
        if query_map.ndim != 4:
            raise ValueError("query_map must be [B,C,H,W]")
        b, c, h, w = query_map.shape
        if c != self.embed_dim:
            raise ValueError(f"Expected {self.embed_dim} channels, got {c}")
        if query_valid.shape == (b, 1, h, w):
            query_valid = query_valid[:, 0]
        if query_valid.shape != (b, h, w):
            raise ValueError("query_valid must be [B,H,W] or [B,1,H,W]")

        query_tokens = query_map.permute(0, 2, 3, 1)
        query = self.q_proj(self.query_norm(query_tokens)).reshape(
            b, h, w, self.num_heads, self.head_dim
        ).permute(0, 3, 4, 1, 2)
        maximum = torch.full(
            (b, self.num_heads, h, w),
            -float("inf"),
            device=query_map.device,
            dtype=torch.float32,
        )
        return {
            "query_map": query_map,
            "query": query,
            "query_valid": query_valid.to(device=query_map.device, dtype=torch.bool),
            "maximum": maximum,
            "denominator": torch.zeros_like(maximum),
            "weighted_value": torch.zeros(
                (b, self.num_heads, self.head_dim, h, w),
                device=query_map.device,
                dtype=torch.float32,
            ),
            "support_count": torch.zeros(
                (b, h, w), device=query_map.device, dtype=torch.float32
            ),
        }

    def accumulate_stream(
        self,
        state: Dict[str, torch.Tensor],
        tail_views: torch.Tensor,
        tail_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Accumulate one arbitrary-size tail chunk into ``state``."""
        if tail_views.ndim != 5:
            raise ValueError("tail_views must be [B,Q,C,H,W]")
        b, q, c, h, w = tail_views.shape
        if c != self.embed_dim:
            raise ValueError(f"Expected {self.embed_dim} tail channels, got {c}")
        if tail_valid.shape != (b, q, h, w):
            raise ValueError("tail_valid must be [B,Q,H,W]")
        if state["query_map"].shape != (b, c, h, w):
            raise ValueError("Tail and R feature shapes must match")

        # Invalid tokens are masked before normalization/projection as well as
        # excluded from softmax, so arbitrary stored values cannot leak in.
        valid = tail_valid.to(device=tail_views.device, dtype=torch.bool)
        base = tail_views.masked_fill(~valid.unsqueeze(2), 0.0)
        tail_tokens = base.permute(0, 1, 3, 4, 2)
        normalized = self.tail_norm(tail_tokens)
        key = self.k_proj(normalized).reshape(
            b, q, h, w, self.num_heads, self.head_dim
        ).permute(0, 1, 4, 5, 2, 3)
        value = self.v_proj(normalized).reshape(
            b, q, h, w, self.num_heads, self.head_dim
        ).permute(0, 1, 4, 5, 2, 3)

        # Pad once, then take spatially shifted views.  The query remains at
        # the centre pixel while each offset supplies a candidate K/V.
        flat_key = key.reshape(b * q, self.attention_dim, h, w)
        flat_value = value.reshape(b * q, self.attention_dim, h, w)
        padded_key = F.pad(flat_key, (self.radius,) * 4).reshape(
            b, q, self.num_heads, self.head_dim,
            h + 2 * self.radius, w + 2 * self.radius,
        )
        padded_value = F.pad(flat_value, (self.radius,) * 4).reshape(
            b, q, self.num_heads, self.head_dim,
            h + 2 * self.radius, w + 2 * self.radius,
        )
        padded_valid = F.pad(
            valid.reshape(b * q, 1, h, w).to(torch.float32),
            (self.radius,) * 4,
        ).reshape(b, q, h + 2 * self.radius, w + 2 * self.radius).bool()

        maximum = state["maximum"]
        denominator = state["denominator"]
        weighted_value = state["weighted_value"]
        neighbourhood_support = torch.zeros_like(valid)
        query = state["query"]
        query_valid = state["query_valid"]
        offset_id = 0
        for oy in range(self.window_size):
            for ox in range(self.window_size):
                shifted_key = padded_key[..., oy:oy + h, ox:ox + w]
                shifted_value = padded_value[..., oy:oy + h, ox:ox + w]
                shifted_valid = padded_valid[..., oy:oy + h, ox:ox + w]
                usable = shifted_valid & query_valid[:, None]
                neighbourhood_support |= shifted_valid

                scores = torch.einsum(
                    "bhdxy,bqhdxy->bqhxy", query.float(), shifted_key.float()
                ) / math.sqrt(self.head_dim)
                scores = scores + self.relative_bias[:, offset_id].reshape(
                    1, 1, self.num_heads, 1, 1
                ).float()
                scores = scores.masked_fill(~usable[:, :, None], -float("inf"))

                candidate_max = scores.max(dim=1).values
                old_has = denominator > 0
                candidate_has = usable.any(dim=1)[:, None].expand(
                    -1, self.num_heads, -1, -1
                )
                new_max = torch.maximum(maximum, candidate_max)
                new_max = torch.where(
                    old_has | candidate_has, new_max, torch.zeros_like(new_max)
                )
                old_factor = torch.where(
                    old_has,
                    torch.exp(maximum - new_max),
                    torch.zeros_like(new_max),
                )
                weights = torch.where(
                    usable[:, :, None],
                    torch.exp(scores - new_max[:, None]),
                    torch.zeros_like(scores),
                )
                weighted_value = (
                    weighted_value * old_factor.unsqueeze(2)
                    + torch.einsum(
                        "bqhxy,bqhdxy->bhdxy", weights, shifted_value.float()
                    )
                )
                denominator = denominator * old_factor + weights.sum(dim=1)
                maximum = new_max
                offset_id += 1

        # A tail is counted once at a query pixel if any location in its local
        # window is valid.  This retains the familiar [0,1] support meaning.
        support_count = state["support_count"] + (
            neighbourhood_support & query_valid[:, None]
        ).sum(dim=1).float()
        return {
            **state,
            "maximum": maximum,
            "denominator": denominator,
            "weighted_value": weighted_value,
            "support_count": support_count,
        }

    def finalize_stream(
        self,
        state: Dict[str, torch.Tensor],
        total_tail_slots: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert an accumulated state into a residual and support map."""
        query_map = state["query_map"]
        b, c, h, w = query_map.shape
        denominator = state["denominator"]
        context = state["weighted_value"] / denominator.clamp_min(
            self.eps
        ).unsqueeze(2)
        context = context.permute(0, 3, 4, 1, 2).reshape(
            b, h, w, self.attention_dim
        ).to(dtype=query_map.dtype)
        attended = self.out_dropout(self.out_proj(context))
        query_tokens = query_map.permute(0, 2, 3, 1)
        candidate = query_tokens + attended
        delta = attended + self.ffn(self.ffn_norm(candidate))

        has_evidence = denominator.sum(dim=1) > 0
        usable = state["query_valid"] & has_evidence
        delta = delta.masked_fill(~usable.unsqueeze(-1), 0.0)
        update = delta.permute(0, 3, 1, 2)
        support = state["support_count"].unsqueeze(1)
        support = support / float(max(int(total_tail_slots), 1))
        support = support.to(dtype=query_map.dtype)
        return update, support

    def forward(
        self,
        query_map: torch.Tensor,
        query_valid: torch.Tensor,
        tail_views: torch.Tensor,
        tail_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.init_stream(query_map, query_valid)
        state = self.accumulate_stream(state, tail_views, tail_valid)
        return self.finalize_stream(state, total_tail_slots=tail_views.shape[1])
