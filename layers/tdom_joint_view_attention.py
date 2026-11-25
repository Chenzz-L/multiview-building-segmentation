"""Flat TDOM-query control for the hierarchical R/tail fusion experiment.

The existing dataset supplies the same fixed anchors and sampled tails.  They
remain separate for TailLocal preprocessing, then become ONE K+Q key/value set
at each spatial position.  No R, anchor pooling, deformable sampling, new gate,
or view-index embedding is introduced.  The final concat projection is the
same operator/initialisation as the Concat reference model.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from layers.tail_local_context import TailLocalContext


SCALES = ("res2", "res3", "res4", "res5")


class TDOMJointViewAttentionFusion(nn.Module):
    def __init__(
        self, embed_dim=256, num_heads=8, num_scales=4, ffn_ratio=2.0,
        dropout=0.0, alpha_init=0.05, eps=1e-6,
        tail_local_context_enabled=True,
        tail_local_context_scales=("res2", "res3"),
        tail_local_context_kernel_size=5,
        tail_local_context_bottleneck_ratio=4,
        tail_local_context_dropout=0.0,
        tail_local_context_alpha_init=1.0,
        tail_local_context_chunk_size=4,
        query_chunk_size=2048,
    ):
        super().__init__()
        if not 1 <= num_scales <= len(SCALES) or embed_dim % num_heads:
            raise ValueError("Invalid scale count or attention head dimension")
        if query_chunk_size < 1 or tail_local_context_chunk_size < 1:
            raise ValueError("Chunk sizes must be positive")
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.final_fusion = 'concat_project'
        self.mask_zero_tail_features = True
        self.eps = eps
        self.query_chunk_size = query_chunk_size
        self.tail_local_context_chunk_size = tail_local_context_chunk_size
        self.tail_local_context_enabled = tail_local_context_enabled
        enabled = tuple(tail_local_context_scales or ()) if tail_local_context_enabled else ()
        if set(enabled) - set(SCALES[:num_scales]):
            raise ValueError("Unknown TailLocal scale")
        self.tail_local_context = nn.ModuleDict({
            scale: TailLocalContext(
                embed_dim, tail_local_context_kernel_size,
                tail_local_context_bottleneck_ratio, tail_local_context_dropout,
                tail_local_context_alpha_init,
            ) for scale in enabled
        })
        self.norm_q = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_scales)])
        self.norm_kv = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_scales)])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_scales)
        ])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_scales)])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * ffn_ratio)), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(int(embed_dim * ffn_ratio), embed_dim),
            nn.Dropout(dropout),
        ) for _ in range(num_scales)])
        # In the reference Concat module alpha is absorbed into the initial
        # image half of the projection and is NOT an additional learned gate.
        self.alpha = nn.Parameter(torch.full((num_scales,), float(alpha_init)), requires_grad=False)
        self.final_projections = nn.ModuleList([
            nn.Conv2d(2 * embed_dim, embed_dim, 1) for _ in range(num_scales)
        ])
        with torch.no_grad():
            eye = torch.eye(embed_dim)
            for projection in self.final_projections:
                projection.weight.zero_()
                projection.weight[:, :embed_dim, 0, 0].copy_(eye)
                projection.weight[:, embed_dim:, 0, 0].copy_(alpha_init * eye)
                projection.bias.zero_()

    def _prepare(self, features, exists, scale, reference):
        b, c, h, w = reference.shape
        if not features:
            return reference.new_zeros(b, 0, c, h, w), torch.zeros(
                b, 0, h, w, device=reference.device, dtype=torch.bool)
        values = []
        for view in features:
            value = view[scale].to(device=reference.device, dtype=reference.dtype)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.shape[:2] != (b, c):
                raise ValueError("Image features must match TDOM batch and channels")
            if value.shape[-2:] != (h, w):
                value = F.interpolate(value, size=(h, w), mode="bilinear", align_corners=False)
            values.append(value)
        stacked = torch.stack(values, dim=1)
        valid = stacked.abs().sum(2) > self.eps
        if exists is not None:
            if tuple(exists.shape) != tuple(stacked.shape[:2]):
                raise ValueError("View-existence mask does not match supplied views")
            valid = valid & exists.to(device=reference.device, dtype=torch.bool)[:, :, None, None]
        return stacked.masked_fill(~valid.unsqueeze(2), 0), valid

    def _tail_context(self, tails, valid, scale, reference):
        if scale not in self.tail_local_context:
            return tails, None, None
        module = self.tail_local_context[scale]
        if tails.shape[1] == 0:
            # Keep trainable parameters in the graph on empty-view DDP ranks.
            dummy, alpha, ratio = module(torch.zeros_like(reference), torch.zeros_like(reference[:, 0], dtype=torch.bool))
            return tails + dummy.sum() * 0, alpha, ratio
        b, q, c, h, w = tails.shape
        pieces, ratios = [], []
        for start in range(0, q, self.tail_local_context_chunk_size):
            count = min(self.tail_local_context_chunk_size, q - start)
            refined, alpha, ratio = module(
                tails[:, start:start + count].reshape(b * count, c, h, w),
                valid[:, start:start + count].reshape(b * count, h, w),
            )
            pieces.append(refined.reshape(b, count, c, h, w))
            ratios.append(ratio)
        return torch.cat(pieces, dim=1), alpha, torch.stack(ratios).mean()

    def forward(
        self, tdom_features, cam_features=None, cam_exist=None, *,
        anchor_features=None, anchor_exist=None, tail_features=None, tail_exist=None,
        output_attn=False, return_aux=False,
    ):
        if len(tdom_features) != self.num_scales:
            raise ValueError("Expected all configured TDOM scales")
        if anchor_features is None:
            anchor_features = cam_features
            anchor_exist = cam_exist if anchor_exist is None else anchor_exist
        fused, maps = [], {}
        aux = {"alpha": [], "joint_support": [], "joint_update": [],
               "tail_local_context_alpha": [], "tail_local_context_delta_ratio": []}
        for sid, (scale, reference) in enumerate(zip(SCALES, tdom_features)):
            anchors, av = self._prepare(anchor_features, anchor_exist, scale, reference)
            tails, tv = self._prepare(tail_features, tail_exist, scale, reference)
            tails, local_alpha, local_ratio = self._tail_context(tails, tv, scale, reference)
            views = torch.cat((anchors, tails), dim=1)
            valid = torch.cat((av, tv), dim=1)
            b, c, h, w = reference.shape
            if views.shape[1] == 0:
                # One masked dummy is a numerical device, never an exposed null
                # candidate; it guarantees finite output and connected gradients.
                views = reference.new_zeros(b, 1, c, h, w) + views.sum() * 0
                valid = torch.zeros(b, 1, h, w, device=reference.device, dtype=torch.bool)
            n = views.shape[1]
            kv = views.permute(0, 3, 4, 1, 2).reshape(b * h * w, n, c)
            flags = valid.permute(0, 2, 3, 1).reshape(b * h * w, n)
            query = reference.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
            updates, weights = [], []
            for start in range(0, len(query), self.query_chunk_size):
                q = query[start:start + self.query_chunk_size]
                keys = kv[start:start + self.query_chunk_size]
                mask = flags[start:start + self.query_chunk_size]
                empty = ~mask.any(1)
                padding = ~mask.clone()
                padding[empty, 0] = False
                norm_keys = self.norm_kv[sid](keys)
                attended, attn_weights = self.attn[sid](
                    self.norm_q[sid](q), norm_keys, norm_keys,
                    key_padding_mask=padding, need_weights=output_attn,
                    average_attn_weights=False,
                )
                delta = attended + self.ffn[sid](self.norm_ffn[sid](q + attended))
                updates.append(delta.masked_fill(empty[:, None, None], 0))
                if output_attn:
                    weights.append(attn_weights.masked_fill(empty[:, None, None, None], 0))
            update = torch.cat(updates).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            fused.append(self.final_projections[sid](torch.cat((reference, update), dim=1)))
            if output_attn:
                maps[scale] = torch.cat(weights)
            if return_aux:
                aux["alpha"].append(self.alpha[sid].detach())
                aux["joint_support"].append(valid.sum(1))
                aux["joint_update"].append(update)
                if local_alpha is not None:
                    aux["tail_local_context_alpha"].append(local_alpha)
                    aux["tail_local_context_delta_ratio"].append(local_ratio)
        if output_attn and return_aux:
            return fused, maps, aux
        if output_attn:
            return fused, maps
        if return_aux:
            return fused, aux
        return fused
