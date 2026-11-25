"""Stable image-only memory fusion for multi-view feature maps."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class MaskedMeanMemoryFusion(nn.Module):
    """Aggregate valid image anchors over the view dimension, then add a small residual.

    ``tdom_features`` is a list of four ``[B,C,H,W]`` tensors. ``cam_features``
    is the existing list-of-dicts representation, one dict per view. A view is
    valid only when both ``cam_exist`` says so and its feature has nonzero
    support at the pixel. No spatial attention or TDOM-dependent scoring is
    performed.
    """

    def __init__(self, embed_dim: int = 256, num_scales: int = 4,
                 alpha_init: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.full((num_scales,), float(alpha_init)))

    @staticmethod
    def _stack_views(cam_features, scale: str, batch_size: int,
                     device: torch.device, dtype: torch.dtype):
        if not cam_features:
            return None
        values = []
        for view in cam_features:
            value = view[scale]
            if value.ndim == 3:
                value = value.unsqueeze(0)
            values.append(value.to(device=device, dtype=dtype))
        return torch.stack(values, dim=1)  # [B,K,C,H,W]

    def forward(self, tdom_features: Sequence[torch.Tensor], cam_features,
                cam_exist: Optional[torch.Tensor] = None,
                output_attn: bool = False, return_aux: bool = False):
        fused = []
        aux = {"memory": [], "support": []}
        for scale_id, (tdom, scale) in enumerate(zip(tdom_features, ("res2", "res3", "res4", "res5"))):
            views = self._stack_views(cam_features, scale, tdom.shape[0], tdom.device, tdom.dtype)
            if views is None:
                memory = torch.zeros_like(tdom)
                support = torch.zeros((tdom.shape[0], 1, *tdom.shape[-2:]), device=tdom.device, dtype=tdom.dtype)
            else:
                valid = views.abs().sum(dim=2, keepdim=True) > self.eps  # [B,K,1,H,W]
                if cam_exist is not None:
                    exists = cam_exist.to(device=tdom.device, dtype=torch.bool)
                    valid = valid & exists[:, :, None, None, None]
                weights = valid.to(dtype=tdom.dtype)
                count = weights.sum(dim=1)
                memory = (views * weights).sum(dim=1) / count.clamp_min(1.0)
                memory = torch.where(count > 0, memory, torch.zeros_like(memory))
                support = (count / max(views.shape[1], 1)).clamp(0.0, 1.0)
            fused.append(tdom + self.alpha[scale_id].to(dtype=tdom.dtype) * memory)
            aux["memory"].append(memory)
            aux["support"].append(support)

        if output_attn and return_aux:
            return fused, None, aux
        if output_attn:
            return fused, None
        if return_aux:
            return fused, aux
        return fused
