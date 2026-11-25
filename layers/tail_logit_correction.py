"""Local, zero-initialized tail correction after a frozen segmentation decoder.

No PPM, top-down FPN, or spatial normalization is used in this head. Each
native-scale pixel sees R, the tail readout, and the frozen coarse prediction.
Only interpolation connects scales; the original decoder never sees tails.
"""
import torch
from torch import nn
import torch.nn.functional as F


class TailLogitCorrection(nn.Module):
    def __init__(self, embed_dim=256, num_scales=4, hidden_dim=64,
                 out_channels=1, gamma_init=0.1):
        super().__init__()
        if min(embed_dim, num_scales, hidden_dim, out_channels) < 1:
            raise ValueError("Correction dimensions must be positive")
        if gamma_init <= 0:
            raise ValueError("gamma_init must be positive (the output is zero-init)")
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * embed_dim + out_channels, hidden_dim, 1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, out_channels, 1),
            ) for _ in range(num_scales)
        ])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, memories, tail_updates, eligible, base_logits):
        n = len(self.heads)
        if not (len(memories) == len(tail_updates) == len(eligible) == n):
            raise ValueError("R, tail updates and eligibility must match all head scales")
        correction = torch.zeros_like(base_logits)
        footprint = torch.zeros_like(base_logits[:, :1], dtype=torch.bool)
        for head, memory, update, valid in zip(
                self.heads, memories, tail_updates, eligible):
            if memory.shape != update.shape or valid.shape != memory[:, :1].shape:
                raise ValueError("Mismatched R/tail/mask shape")
            coarse = F.interpolate(base_logits.detach(), size=memory.shape[-2:],
                                   mode="bilinear", align_corners=False)
            delta = head(torch.cat((memory.detach(), update, coarse), dim=1))
            # Mask before AND after interpolation. Padding/all-zero tails must
            # not create logits through biases or interpolation of neighbours.
            delta = delta.masked_fill(~valid, 0)
            dense_valid = F.interpolate(valid.float(), size=base_logits.shape[-2:],
                                        mode="nearest").bool()
            delta = F.interpolate(delta, size=base_logits.shape[-2:],
                                  mode="bilinear", align_corners=False)
            correction = correction + delta.masked_fill(~dense_valid, 0) / n
            footprint = footprint | dense_valid
        correction = self.gamma.to(correction.dtype) * correction
        return base_logits + correction, {
            "tail_logit_gamma": [self.gamma.detach()],
            "tail_logit_abs_mean": [correction.detach().float().abs().mean()],
            "tail_logit_eligible_fraction": [footprint.float().mean()],
        }
