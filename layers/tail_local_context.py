"""Lightweight local-context encoder for individual projected tail features.

The module is deliberately independent for every tail view.  It does not mix
views and does not add another attention stage: a depthwise convolution carries
relative local-position information, while a small bottleneck mixes channels.
The valid-footprint ratio is injected explicitly so projected-feature coverage
boundaries are distinguishable from ordinary zero-valued neighbourhoods.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    """Use the largest stable GroupNorm group count that divides channels."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TailLocalContext(nn.Module):
    """Mask-aware depthwise residual local context for one projected tail map.

    ``valid_map`` is ``[B,H,W]`` and follows the projected-feature convention:
    invalid pixels must neither receive nor contribute local context.  The last
    projection is zero-initialised, therefore ``forward`` is an exact identity
    at initialisation even with ``alpha_init=1``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 5,
        bottleneck_ratio: int = 4,
        dropout: float = 0.0,
        alpha_init: float = 1.0,
    ):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("tail local-context kernel_size must be odd and >= 3")
        if bottleneck_ratio < 1:
            raise ValueError("tail local-context bottleneck_ratio must be >= 1")
        hidden = max(embed_dim // int(bottleneck_ratio), 16)
        self.kernel_size = int(kernel_size)
        self.depthwise = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=embed_dim,
            bias=False,
        )
        self.down = nn.Conv2d(embed_dim, hidden, kernel_size=1, bias=False)
        # An explicit valid-footprint statistic lets the same feature value be
        # interpreted differently at a coverage boundary and in its interior.
        self.support_proj = nn.Conv2d(1, hidden, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(_group_count(hidden), hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Conv2d(hidden, embed_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self,
        tail: torch.Tensor,
        valid_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tail.ndim != 4:
            raise ValueError(f"TailLocalContext expects [B,C,H,W], got {tuple(tail.shape)}")
        if valid_map is None:
            valid_map = torch.ones(
                tail.shape[0], tail.shape[-2], tail.shape[-1],
                dtype=torch.bool, device=tail.device,
            )
        if valid_map.shape != (tail.shape[0], tail.shape[-2], tail.shape[-1]):
            raise ValueError(
                "valid_map must be [B,H,W] matching the tail feature map"
            )

        valid = valid_map.unsqueeze(1).to(dtype=tail.dtype)
        masked_tail = tail * valid
        support = F.avg_pool2d(
            valid,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        hidden = self.depthwise(masked_tail)
        hidden = self.down(hidden) + self.support_proj(support)
        hidden = self.dropout(self.activation(self.norm(hidden)))
        delta = self.up(hidden) * valid
        refined = (tail + self.alpha.to(dtype=tail.dtype) * delta) * valid
        ratio = delta.float().norm() / masked_tail.float().norm().clamp_min(1.0e-6)
        return refined, self.alpha.detach(), ratio.detach()
