"""SAIL-style image-only memory fusion.

This module deliberately keeps the multi-view interaction on the image/view
axis.  It does *not* use a TDOM feature as a query. Selected anchor views are
first contextualised with each other. The direction of the following
cross-attention is configurable: either every tail independently queries the
anchors, or the compact anchor memory queries all tails. Tail views never use
one another as queries.

The public input format follows the existing fusion layers:

``list[dict[res2/res3/res4/res5 -> [B, C, H, W]]]`` plus ``[B, N]`` view
existence flags.  Until the Dataset emits separate anchors and tails, passing
only ``cam_features`` is supported and treats all supplied views as anchors.
When available, callers may pass ``anchor_features`` / ``tail_features`` and
their corresponding ``*_exist`` flags as keyword arguments.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from layers.swin_r import SwinRRefiner
from layers.multi_latent_swin_r import MultiLatentSwinR
from layers.single_r_swin_set import SingleRSwinSet
from layers.tail_local_context import TailLocalContext
from layers.r_conditioned_local_tail_attention import (
    RConditionedLocalTailAttention,
)
from layers.two_token_memory import (
    ThreeWayGatedTailReadout,
    TwoTokenTailReadout,
)
from layers.dual_memory_tail_calibrator import DualMemoryTailCalibrator


_SCALES = ("res2", "res3", "res4", "res5")


class _ViewAxisTransformerBlock(nn.Module):
    """A pre-norm transformer block operating only across views at one pixel."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float,
                 ffn_ratio: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        hidden_dim = int(embed_dim * ffn_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Args: tokens ``[N, K, C]``; valid ``[N, K]``."""
        key_padding_mask, all_invalid = _safe_key_padding_mask(valid)
        attended, _ = self.attn(
            self.norm1(tokens), self.norm1(tokens), self.norm1(tokens),
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        output = tokens + attended
        output = output + self.ffn(self.norm2(output))
        output = output.masked_fill(~valid.unsqueeze(-1), 0.0)
        if all_invalid.any():
            output[all_invalid] = 0.0
        return output


class _CrossViewLocalAttentionBlock(nn.Module):
    """Spatially local attention over an unordered set of anchor views.

    A query token from anchor ``i`` at ground location ``(x, y)`` attends to
    every valid anchor ``j`` in a small window centred at ``(x, y)``.  This is
    deliberately *not* a copied Swin block: there are no artificial window
    partitions and no view-index embeddings.  Sliding windows let it handle
    small projection offsets, while omitting a view embedding keeps the block
    equivariant to the arbitrary order of K-medoid anchors.

    The attention softmax is reduced one spatial offset at a time.  That is
    materially lighter than materialising ``[B, H*W, K, K*W*W, C]`` at res3.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        ffn_ratio: float,
        residual_init: float,
    ):
        super().__init__()
        if window_size < 3 or window_size % 2 != 1:
            raise ValueError("anchor spatial window_size must be an odd integer >= 3")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.window_size = int(window_size)
        self.radius = window_size // 2
        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        # Offset-only bias, shared by every anchor pair.  A view-id embedding
        # would make the result depend on the arbitrary order of medoids.
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size)
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        # Start as a conservative residual change to a pretrained feature map.
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(self, views: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Args: ``views [B,K,C,H,W]`` and ``valid [B,K,H,W]``."""
        b, k, c, h, w = views.shape
        if valid.shape != (b, k, h, w):
            raise ValueError(
                f"valid must be {(b, k, h, w)}, got {tuple(valid.shape)}"
            )
        # Invalid projections are absent keys/values, not zero-valued evidence.
        base = views.masked_fill(~valid.unsqueeze(2), 0.0)
        tokens = base.permute(0, 1, 3, 4, 2)  # [B,K,H,W,C]
        normed = self.norm1(tokens)
        query = self.q_proj(normed).reshape(
            b, k, h, w, self.num_heads, self.head_dim
        ).permute(0, 1, 4, 5, 2, 3)  # [B,K,heads,head_dim,H,W]

        # Pad once and then scan offsets.  Online softmax avoids a large
        # K-by-(K*window^2) attention tensor at every ground location.
        # Project K/V once for the full feature map, then shift the projected
        # maps.  Re-projecting them for every offset would make a 5x5 window
        # roughly 25 times more expensive than necessary.
        key_map = self.k_proj(normed).permute(0, 1, 4, 2, 3).reshape(b * k, c, h, w)
        value_map = self.v_proj(normed).permute(0, 1, 4, 2, 3).reshape(b * k, c, h, w)
        padded_key = F.pad(key_map, (self.radius,) * 4).reshape(
            b, k, c, h + 2 * self.radius, w + 2 * self.radius
        )
        padded_value = F.pad(value_map, (self.radius,) * 4).reshape(
            b, k, c, h + 2 * self.radius, w + 2 * self.radius
        )
        padded_valid = F.pad(
            valid.reshape(b * k, 1, h, w).to(dtype=torch.float32),
            (self.radius,) * 4,
        ).reshape(b, k, h + 2 * self.radius, w + 2 * self.radius).bool()

        work_dtype = torch.float32
        maximum = torch.full(
            (b, k, self.num_heads, h, w), -float("inf"),
            device=views.device, dtype=work_dtype,
        )
        denominator = torch.zeros_like(maximum)
        weighted_values = torch.zeros(
            (b, k, self.num_heads, self.head_dim, h, w),
            device=views.device, dtype=work_dtype,
        )
        offset_id = 0
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                key = padded_key[
                    :, :, :, self.radius + dy:self.radius + dy + h,
                    self.radius + dx:self.radius + dx + w,
                ].reshape(b, k, self.num_heads, self.head_dim, h, w)
                value = padded_value[
                    :, :, :, self.radius + dy:self.radius + dy + h,
                    self.radius + dx:self.radius + dx + w,
                ].reshape(b, k, self.num_heads, self.head_dim, h, w)
                source_valid = padded_valid[
                    :, :, self.radius + dy:self.radius + dy + h,
                    self.radius + dx:self.radius + dx + w,
                ]

                # i is the query-anchor and j is the key/value-anchor.
                scores = torch.einsum(
                    "bihdxy,bjhdxy->bijhxy", query.float(), key.float()
                ) / math.sqrt(self.head_dim)
                scores = scores + self.relative_bias[:, offset_id].view(
                    1, 1, 1, self.num_heads, 1, 1
                ).float()
                source_mask = source_valid[:, None, :, None, :, :]
                scores = scores.masked_fill(~source_mask, -float("inf"))

                offset_max = scores.max(dim=2).values
                new_maximum = torch.maximum(maximum, offset_max)
                old_weight = torch.where(
                    torch.isfinite(maximum),
                    torch.exp(maximum - new_maximum),
                    torch.zeros_like(maximum),
                )
                weights = torch.where(
                    source_mask,
                    torch.exp(scores - new_maximum.unsqueeze(2)),
                    torch.zeros_like(scores),
                )
                value_sum = torch.einsum(
                    "bijhxy,bjhdxy->bihdxy", weights, value.float()
                )
                denominator = denominator * old_weight + weights.sum(dim=2)
                weighted_values = (
                    weighted_values * old_weight.unsqueeze(3) + value_sum
                )
                maximum = new_maximum
                offset_id += 1

        attended = weighted_values / denominator.clamp_min(1e-12).unsqueeze(3)
        attended = torch.where(
            denominator.unsqueeze(3) > 0, attended, torch.zeros_like(attended)
        )
        attended = attended.to(dtype=views.dtype).permute(0, 1, 4, 5, 2, 3)
        attended = attended.reshape(b, k, h, w, c)
        attention_delta = self.out_proj(attended)
        candidate = tokens + attention_delta
        delta = attention_delta + self.ffn(self.norm2(candidate))
        output = tokens + self.residual_scale.to(dtype=tokens.dtype) * delta
        output = output.masked_fill(~valid.unsqueeze(-1), 0.0)
        return output.permute(0, 1, 4, 2, 3)


class _TailQueryLocalAnchorMemory(nn.Module):
    """Let each tail token read an unpooled local anchor-token memory set.

    ``tail[q, x, y]`` attends to all valid ``anchor[k, x+dx, y+dy]``.  Unlike
    the old ``MultiheadAttention`` path, anchor tokens are not first reduced
    with a masked mean.  Tail views remain independent, so their calibrated
    outputs can still be accumulated chunk-by-chunk for all-tail inference.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        ffn_ratio: float,
        residual_init: float,
    ):
        super().__init__()
        if window_size < 3 or window_size % 2 != 1:
            raise ValueError("tail-to-anchor local window_size must be odd and >= 3")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.window_size = int(window_size)
        self.radius = window_size // 2
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size)
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(
        self,
        tails: torch.Tensor,
        tail_valid: torch.Tensor,
        anchors: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return calibrated tails and their usable map.

        Args:
            tails: ``[B,Q,C,H,W]``; anchors: ``[B,K,C,H,W]``.
            ``*_valid`` are corresponding ``[B,Q/H,K/H,W]`` boolean maps.
        """
        b, q, c, h, w = tails.shape
        ba, k, ca, ha, wa = anchors.shape
        if (ba, ca, ha, wa) != (b, c, h, w):
            raise ValueError("Tail and anchor maps must have matching B/C/H/W")
        if tail_valid.shape != (b, q, h, w) or anchor_valid.shape != (b, k, h, w):
            raise ValueError("Invalid tail/anchor validity-map shape")

        tail_tokens = tails.masked_fill(~tail_valid.unsqueeze(2), 0.0).permute(0, 1, 3, 4, 2)
        anchor_tokens = anchors.masked_fill(~anchor_valid.unsqueeze(2), 0.0).permute(0, 1, 3, 4, 2)
        query = self.q_proj(self.norm_q(tail_tokens)).reshape(
            b, q, h, w, self.num_heads, self.head_dim
        ).permute(0, 1, 4, 5, 2, 3)
        anchor_normed = self.norm_kv(anchor_tokens)
        key_map = self.k_proj(anchor_normed).permute(0, 1, 4, 2, 3).reshape(b * k, c, h, w)
        value_map = self.v_proj(anchor_normed).permute(0, 1, 4, 2, 3).reshape(b * k, c, h, w)
        padded_key = F.pad(key_map, (self.radius,) * 4).reshape(
            b, k, c, h + 2 * self.radius, w + 2 * self.radius
        )
        padded_value = F.pad(value_map, (self.radius,) * 4).reshape(
            b, k, c, h + 2 * self.radius, w + 2 * self.radius
        )
        padded_valid = F.pad(
            anchor_valid.reshape(b * k, 1, h, w).to(dtype=torch.float32),
            (self.radius,) * 4,
        ).reshape(b, k, h + 2 * self.radius, w + 2 * self.radius).bool()

        maximum = torch.full(
            (b, q, self.num_heads, h, w), -float("inf"),
            device=tails.device, dtype=torch.float32,
        )
        denominator = torch.zeros_like(maximum)
        weighted_values = torch.zeros(
            (b, q, self.num_heads, self.head_dim, h, w),
            device=tails.device, dtype=torch.float32,
        )
        offset_id = 0
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                key = padded_key[
                    :, :, :, self.radius + dy:self.radius + dy + h,
                    self.radius + dx:self.radius + dx + w,
                ].reshape(b, k, self.num_heads, self.head_dim, h, w)
                value = padded_value[
                    :, :, :, self.radius + dy:self.radius + dy + h,
                    self.radius + dx:self.radius + dx + w,
                ].reshape(b, k, self.num_heads, self.head_dim, h, w)
                source_valid = padded_valid[
                    :, :, self.radius + dy:self.radius + dy + h,
                    self.radius + dx:self.radius + dx + w,
                ]
                scores = torch.einsum(
                    "bqhdxy,bkhdxy->bqkhxy", query.float(), key.float()
                ) / math.sqrt(self.head_dim)
                scores = scores + self.relative_bias[:, offset_id].view(
                    1, 1, 1, self.num_heads, 1, 1
                ).float()
                source_mask = source_valid[:, None, :, None, :, :]
                scores = scores.masked_fill(~source_mask, -float("inf"))
                offset_max = scores.max(dim=2).values
                new_maximum = torch.maximum(maximum, offset_max)
                old_weight = torch.where(
                    torch.isfinite(maximum),
                    torch.exp(maximum - new_maximum),
                    torch.zeros_like(maximum),
                )
                weights = torch.where(
                    source_mask,
                    torch.exp(scores - new_maximum.unsqueeze(2)),
                    torch.zeros_like(scores),
                )
                value_sum = torch.einsum(
                    "bqkhxy,bkhdxy->bqhdxy", weights, value.float()
                )
                denominator = denominator * old_weight + weights.sum(dim=2)
                weighted_values = weighted_values * old_weight.unsqueeze(3) + value_sum
                maximum = new_maximum
                offset_id += 1

        attended = weighted_values / denominator.clamp_min(1e-12).unsqueeze(3)
        usable = tail_valid & (denominator.sum(dim=2) > 0)
        attended = torch.where(
            usable.unsqueeze(2).unsqueeze(3), attended, torch.zeros_like(attended)
        )
        attended = attended.to(dtype=tails.dtype).permute(0, 1, 4, 5, 2, 3).reshape(
            b, q, h, w, c
        )
        attention_delta = self.out_proj(attended)
        candidate = tail_tokens + attention_delta
        delta = attention_delta + self.ffn(self.norm_ffn(candidate))
        calibrated = tail_tokens + self.residual_scale.to(dtype=tails.dtype) * delta
        calibrated = calibrated.masked_fill(~usable.unsqueeze(-1), 0.0)
        return calibrated.permute(0, 1, 4, 2, 3), usable


def _safe_key_padding_mask(valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build an MHA key mask without creating NaNs for empty pixels.

    PyTorch attention returns NaN when every key of a sequence is masked.  We
    temporarily expose one zero token for those rows and zero their outputs
    afterwards.  ``valid`` is true for usable tokens.
    """
    if valid.ndim != 2:
        raise ValueError(f"valid must be [N, K], got {tuple(valid.shape)}")
    key_padding_mask = ~valid
    all_invalid = ~valid.any(dim=1)
    if all_invalid.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_invalid, 0] = False
    return key_padding_mask, all_invalid


class SAILImageMemoryFusion(nn.Module):
    """Construct an anchor representation and calibrate independent tails.

    Per FPN scale and ground pixel, the computation is:

    ``R = AnchorSelfAttention(A)``

    Optionally, selected FPN scales first use a cross-view local-attention
    block.  It lets an anchor feature at ``(x,y)`` compare the K anchors in a
    small spatial neighbourhood before the usual view-axis consensus and
    masked mean form ``R``.
    With ``attention_direction='tail_to_anchor'`` (the original baseline):

    ``T' = Tail + CrossAttention(Q=Tail, K/V=R)``
    ``M = mean(R) + beta * mean(T')``

    With ``attention_direction='anchor_to_tail'`` (the R-query experiment):

    ``dR = CrossAttention(Q=mean(R), K/V=Tail) + FFN``
    ``M = mean(R) + beta * dR``

    ``F = TDOM + alpha * M``

    Invalid projected pixels are excluded from all means and attention keys.
    ``beta`` starts at zero, so a new model starts as an anchor-only memory
    baseline while retaining a learnable path for all tail views.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_scales: int = 4,
        anchor_depth: int = 1,
        dropout: float = 0.0,
        ffn_ratio: float = 2.0,
        alpha_init: float = 0.05,
        tail_beta_init: float = 0.0,
        attention_direction: str = "tail_to_anchor",
        use_null_tail: bool = False,
        null_tail_logit_init: float = -1.0,
        tail_aggregation: str = "attention",
        tail_topk: int = 8,
        tail_topk_min: int = 1,
        tail_topk_eval: Optional[int] = None,
        tail_topk_random_train: bool = False,
        tail_global_score_scale: str = "res3",
        tail_score_temperature: float = 1.0,
        anchor_spatial_window_size: int = 1,
        anchor_spatial_scales: Optional[Sequence[str]] = None,
        anchor_spatial_depth: int = 1,
        anchor_spatial_residual_init: float = 0.1,
        swin_r_enabled: bool = False,
        swin_r_scales: Optional[Sequence[str]] = None,
        swin_r_window_size: int = 8,
        swin_r_num_heads: int = 8,
        swin_r_depth: int = 2,
        swin_r_mlp_ratio: float = 4.0,
        swin_r_dropout: float = 0.0,
        swin_r_alpha_init: float = 0.01,
        multi_latent_swin_r_enabled: bool = False,
        multi_latent_num: int = 3,
        multi_latent_spatial_window_size: int = 8,
        multi_latent_spatial_depth: int = 2,
        multi_latent_depth: int = 1,
        multi_latent_beta_init: float = 0.01,
        multi_latent_alpha_init: float = 0.01,
        single_r_swin_set_enabled: bool = False,
        single_r_swin_set_scales: Optional[Sequence[str]] = None,
        single_r_spatial_window_size: int = 8,
        single_r_spatial_depth: int = 2,
        single_r_view_depth: int = 1,
        single_r_beta_init: float = 0.01,
        single_r_spatial_alpha_init: float = 0.01,
        tail_local_context_enabled: bool = False,
        tail_local_context_scales: Optional[Sequence[str]] = None,
        tail_local_context_kernel_size: int = 5,
        tail_local_context_bottleneck_ratio: int = 4,
        tail_local_context_dropout: float = 0.0,
        tail_local_context_alpha_init: float = 1.0,
        tail_local_context_chunk_size: int = 4,
        r_conditioned_local_tail_enabled: bool = False,
        r_conditioned_local_tail_scales: Optional[Sequence[str]] = None,
        r_conditioned_local_tail_window_sizes: Optional[Dict[str, int]] = None,
        r_conditioned_local_tail_attention_dim: int = 64,
        r_conditioned_local_tail_num_heads: int = 4,
        r_conditioned_local_tail_ffn_ratio: float = 2.0,
        r_conditioned_local_tail_dropout: float = 0.0,
        tail_dual_memory_context_enabled: bool = False,
        tail_dual_memory_context_scales: Optional[Sequence[str]] = None,
        tail_dual_memory_context_ffn_ratio: float = 2.0,
        tail_dual_memory_context_alpha_init: float = 0.01,
        tail_anchor_memory_mode: str = "same_pixel",
        tail_anchor_local_window_size: int = 1,
        tail_anchor_local_scales: Optional[Sequence[str]] = None,
        tail_anchor_local_residual_init: float = 0.1,
        tail_anchor_local_query_chunk_size: int = 4,
        tail_anchor_local_checkpoint: bool = True,
        tdom_memory_token_enabled: bool = False,
        tdom_memory_token_scales: Optional[Sequence[str]] = None,
        tdom_memory_token_hidden_ratio: float = 1.0,
        tdom_memory_token_readout: str = "mlp",
        tdom_memory_gate_hidden_dim: int = 32,
        tdom_in_anchor_memory: bool = False,
        tdom_in_anchor_memory_scales: Optional[Sequence[str]] = None,
        tdom_in_anchor_memory_alpha_init: float = 0.01,
        memory_output_mode: str = "anchor_plus_tail",
        mask_zero_tail_features: bool = True,
        eps: float = 1e-6,
        tail_content_gamma: float = 1.0,
        tail_content_gamma_scales: Optional[Sequence[str]] = None,
        tail_value_mode: str = "projected",
        final_fusion: str = "add",
    ):
        super().__init__()
        if tail_value_mode not in {"projected", "raw_weighted"}:
            raise ValueError("tail_value_mode must be projected or raw_weighted")
        if tail_value_mode == "raw_weighted" and (
            attention_direction != "anchor_to_tail" or tail_aggregation != "attention"
            or not mask_zero_tail_features or use_null_tail
            or tdom_memory_token_enabled or tail_dual_memory_context_enabled
            or r_conditioned_local_tail_enabled or tail_content_gamma != 1
            or dropout != 0
        ):
            raise ValueError(
                "raw_weighted requires masked single-R anchor_to_tail attention, "
                "no null/dual/local reader/content scaling, and dropout=0"
            )
        self.tail_value_mode = tail_value_mode
        if not math.isfinite(tail_content_gamma) or not 0 <= tail_content_gamma <= 1:
            raise ValueError("tail_content_gamma must be finite and in [0, 1]")
        if tail_content_gamma_scales is None:
            tail_content_gamma_scales = ("res2",) if tail_content_gamma != 1 else ()
        if set(tail_content_gamma_scales) - set(_SCALES[:num_scales]):
            raise ValueError("tail_content_gamma_scales contains an unavailable scale")
        if tail_content_gamma != 1 and (
            attention_direction != "anchor_to_tail"
            or tail_aggregation != "attention"
            or not mask_zero_tail_features
            or use_null_tail
            or tdom_memory_token_enabled
            or r_conditioned_local_tail_enabled
        ):
            raise ValueError(
                "tail content decomposition requires masked anchor_to_tail "
                "attention without null, dual-query or R-conditioned local readers"
            )
        # Python attributes only: preserve old checkpoint state_dict keys.
        # gamma=1 follows exactly the historical path and incurs no extra FFN.
        self.tail_content_gamma = float(tail_content_gamma)
        self.tail_content_gamma_scales = frozenset(tail_content_gamma_scales)
        if anchor_depth < 0:
            raise ValueError("anchor_depth must be >= 0")
        if num_scales > len(_SCALES):
            raise ValueError(f"num_scales must be <= {len(_SCALES)}")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if attention_direction not in {"tail_to_anchor", "anchor_to_tail"}:
            raise ValueError(
                "attention_direction must be 'tail_to_anchor' or 'anchor_to_tail'"
            )
        if not mask_zero_tail_features and (
            attention_direction != "anchor_to_tail"
            or tail_aggregation != "attention"
            or use_null_tail
            or tdom_memory_token_enabled
        ):
            raise ValueError(
                "mask_zero_tail_features=false currently requires "
                "anchor_to_tail attention, tail_aggregation='attention', "
                "use_null_tail=false, and tdom_memory_token_enabled=false"
            )
        if tail_aggregation not in {
            "attention", "tailwise_topk", "global_tail_topk", "direct_mean"
        }:
            raise ValueError(
                "tail_aggregation must be 'attention', 'tailwise_topk', "
                "'global_tail_topk', or 'direct_mean'"
            )
        if tail_aggregation == "direct_mean" and attention_direction != "anchor_to_tail":
            raise ValueError(
                "direct_mean requires attention_direction='anchor_to_tail'"
            )
        if tail_aggregation == "direct_mean" and (
            use_null_tail
            or tail_local_context_enabled
            or r_conditioned_local_tail_enabled
            or tail_dual_memory_context_enabled
            or tdom_memory_token_enabled
        ):
            raise ValueError(
                "direct_mean bypasses all tail attention/calibration modules; "
                "disable null-tail, TailLocal, R-conditioned local tail, "
                "dual-memory tail context, and TDOM memory tokens"
            )
        if tail_topk < 1:
            raise ValueError("tail_topk must be >= 1")
        if (
            tail_aggregation == "global_tail_topk"
            and attention_direction != "anchor_to_tail"
        ):
            raise ValueError(
                "global_tail_topk requires attention_direction='anchor_to_tail'"
            )
        if tail_topk_min < 1 or tail_topk_min > tail_topk:
            raise ValueError("tail_topk_min must satisfy 1 <= min <= tail_topk")
        if tail_topk_eval is not None and (
            tail_topk_eval < 1 or tail_topk_eval > tail_topk
        ):
            raise ValueError(
                "tail_topk_eval must satisfy 1 <= eval <= tail_topk"
            )
        if tail_global_score_scale not in _SCALES[:num_scales]:
            raise ValueError(
                f"tail_global_score_scale must be one of {_SCALES[:num_scales]}"
            )
        if tail_score_temperature <= 0:
            raise ValueError("tail_score_temperature must be > 0")
        if anchor_spatial_window_size != 1 and (
            anchor_spatial_window_size < 3 or anchor_spatial_window_size % 2 != 1
        ):
            raise ValueError(
                "anchor_spatial_window_size must be 1 (disabled) or an odd integer >= 3"
            )
        if anchor_spatial_depth < 1:
            raise ValueError("anchor_spatial_depth must be >= 1")
        if tail_anchor_memory_mode not in {"same_pixel", "local_token_set"}:
            raise ValueError(
                "tail_anchor_memory_mode must be 'same_pixel' or 'local_token_set'"
            )
        if tail_anchor_local_window_size != 1 and (
            tail_anchor_local_window_size < 3 or tail_anchor_local_window_size % 2 != 1
        ):
            raise ValueError(
                "tail_anchor_local_window_size must be 1 or an odd integer >= 3"
            )
        if memory_output_mode not in {"anchor_plus_tail", "tail_conditioned", "r_plus_attention"}:
            raise ValueError(
                "memory_output_mode must be anchor_plus_tail, tail_conditioned or r_plus_attention"
            )
        if memory_output_mode == "r_plus_attention" and (
            attention_direction != "anchor_to_tail" or tail_aggregation != "attention"
            or tail_value_mode != "projected" or not mask_zero_tail_features
            or use_null_tail or tdom_memory_token_enabled or tdom_in_anchor_memory
            or tail_dual_memory_context_enabled or r_conditioned_local_tail_enabled
            or tail_content_gamma != 1
        ):
            raise ValueError("r_plus_attention requires plain masked R-query projected tail attention")
        if tail_anchor_local_query_chunk_size < 1:
            raise ValueError("tail_anchor_local_query_chunk_size must be >= 1")
        if tail_local_context_kernel_size < 3 or tail_local_context_kernel_size % 2 != 1:
            raise ValueError(
                "tail_local_context_kernel_size must be an odd integer >= 3"
            )
        if tail_local_context_bottleneck_ratio < 1:
            raise ValueError("tail_local_context_bottleneck_ratio must be >= 1")
        if tail_local_context_chunk_size < 1:
            raise ValueError("tail_local_context_chunk_size must be >= 1")
        if r_conditioned_local_tail_attention_dim < 1:
            raise ValueError("r_conditioned_local_tail_attention_dim must be >= 1")
        if (
            r_conditioned_local_tail_num_heads < 1
            or r_conditioned_local_tail_attention_dim
            % r_conditioned_local_tail_num_heads != 0
        ):
            raise ValueError(
                "r_conditioned_local_tail_attention_dim must be divisible by "
                "r_conditioned_local_tail_num_heads"
            )
        if r_conditioned_local_tail_ffn_ratio <= 0:
            raise ValueError("r_conditioned_local_tail_ffn_ratio must be > 0")
        if tail_dual_memory_context_ffn_ratio <= 0:
            raise ValueError("tail_dual_memory_context_ffn_ratio must be > 0")
        if tail_dual_memory_context_alpha_init < 0:
            raise ValueError("tail_dual_memory_context_alpha_init must be >= 0")
        if tdom_memory_token_hidden_ratio <= 0:
            raise ValueError("tdom_memory_token_hidden_ratio must be > 0")
        if tdom_memory_token_readout not in {"mlp", "three_way_gate"}:
            raise ValueError(
                "tdom_memory_token_readout must be 'mlp' or 'three_way_gate'"
            )
        if tdom_memory_gate_hidden_dim < 1:
            raise ValueError("tdom_memory_gate_hidden_dim must be >= 1")
        if tdom_in_anchor_memory_alpha_init < 0:
            raise ValueError("tdom_in_anchor_memory_alpha_init must be >= 0")
        if anchor_spatial_scales is None:
            anchor_spatial_scales = ()
        unknown_spatial_scales = set(anchor_spatial_scales) - set(_SCALES[:num_scales])
        if unknown_spatial_scales:
            raise ValueError(
                f"Unknown/unavailable anchor_spatial_scales: {sorted(unknown_spatial_scales)}"
            )
        if tail_anchor_local_scales is None:
            tail_anchor_local_scales = ()
        unknown_tail_local_scales = set(tail_anchor_local_scales) - set(_SCALES[:num_scales])
        if unknown_tail_local_scales:
            raise ValueError(
                "Unknown/unavailable tail_anchor_local_scales: "
                f"{sorted(unknown_tail_local_scales)}"
            )
        if tail_anchor_memory_mode == "local_token_set" and tail_anchor_local_window_size == 1:
            raise ValueError(
                "local_token_set requires tail_anchor_local_window_size >= 3"
            )
        if tdom_memory_token_enabled and (
            attention_direction != "anchor_to_tail"
            or tail_aggregation != "attention"
            or use_null_tail
        ):
            raise ValueError(
                "tdom_memory_token requires anchor_to_tail attention, "
                "tail_aggregation='attention', and use_null_tail=false"
            )

        self.eps = float(eps)
        self.num_scales = int(num_scales)
        self.attention_direction = attention_direction
        self.tail_aggregation = tail_aggregation
        self.tail_topk = int(tail_topk)
        self.tail_topk_min = int(tail_topk_min)
        self.tail_topk_eval = int(
            tail_topk if tail_topk_eval is None else tail_topk_eval
        )
        self.tail_topk_random_train = bool(tail_topk_random_train)
        self.tail_global_score_scale = str(tail_global_score_scale)
        self.tail_global_score_scale_id = _SCALES.index(
            self.tail_global_score_scale
        )
        self.tail_score_temperature = float(tail_score_temperature)
        self.anchor_spatial_window_size = int(anchor_spatial_window_size)
        self.anchor_spatial_scales = frozenset(anchor_spatial_scales)
        if swin_r_scales is None:
            swin_r_scales = ("res3", "res4") if swin_r_enabled else ()
        unknown_swin_r_scales = set(swin_r_scales) - set(_SCALES[:num_scales])
        if unknown_swin_r_scales:
            raise ValueError(
                f"Unknown/unavailable swin_r_scales: {sorted(unknown_swin_r_scales)}"
            )
        self.swin_r_enabled = bool(swin_r_enabled)
        self.swin_r_scales = frozenset(swin_r_scales if swin_r_enabled else ())
        if multi_latent_swin_r_enabled and multi_latent_num < 1:
            raise ValueError("multi_latent_num must be >= 1")
        self.multi_latent_swin_r_enabled = bool(multi_latent_swin_r_enabled)
        self.multi_latent_scales = frozenset(
            ("res3", "res4") if multi_latent_swin_r_enabled else ()
        )
        if single_r_swin_set_enabled and multi_latent_swin_r_enabled:
            raise ValueError("single-R and multi-latent R cannot be enabled together")
        if single_r_swin_set_scales is None:
            single_r_swin_set_scales = ("res3",) if single_r_swin_set_enabled else ()
        unknown_single_r_scales = set(single_r_swin_set_scales) - set(_SCALES[:num_scales])
        if unknown_single_r_scales:
            raise ValueError(
                f"Unknown/unavailable single_r_swin_set_scales: {sorted(unknown_single_r_scales)}"
            )
        self.single_r_swin_set_scales = frozenset(
            single_r_swin_set_scales if single_r_swin_set_enabled else ()
        )
        if tail_local_context_scales is None:
            tail_local_context_scales = (
                ("res3",) if tail_local_context_enabled else ()
            )
        unknown_tail_context_scales = set(tail_local_context_scales) - set(
            _SCALES[:num_scales]
        )
        if unknown_tail_context_scales:
            raise ValueError(
                "Unknown/unavailable tail_local_context_scales: "
                f"{sorted(unknown_tail_context_scales)}"
            )
        self.tail_local_context_enabled = bool(tail_local_context_enabled)
        self.tail_local_context_scales = frozenset(
            tail_local_context_scales if tail_local_context_enabled else ()
        )
        self.tail_local_context_chunk_size = int(tail_local_context_chunk_size)
        if r_conditioned_local_tail_enabled and (
            attention_direction != "anchor_to_tail"
            or tail_aggregation != "attention"
            or use_null_tail
            or not mask_zero_tail_features
            or tdom_memory_token_enabled
        ):
            raise ValueError(
                "R-conditioned local tail attention requires anchor_to_tail, "
                "tail_aggregation='attention', use_null_tail=false, "
                "mask_zero_tail_features=true, and no TDOM memory token"
            )
        if r_conditioned_local_tail_scales is None:
            r_conditioned_local_tail_scales = (
                ("res2", "res3") if r_conditioned_local_tail_enabled else ()
            )
        unknown_r_conditioned_scales = set(r_conditioned_local_tail_scales) - set(
            _SCALES[:num_scales]
        )
        if unknown_r_conditioned_scales:
            raise ValueError(
                "Unknown/unavailable r_conditioned_local_tail_scales: "
                f"{sorted(unknown_r_conditioned_scales)}"
            )
        self.r_conditioned_local_tail_scales = frozenset(
            r_conditioned_local_tail_scales
            if r_conditioned_local_tail_enabled else ()
        )
        overlap = self.r_conditioned_local_tail_scales & self.tail_local_context_scales
        if overlap:
            raise ValueError(
                "R-conditioned local tail attention replaces TailLocal at a scale; "
                f"the scale sets must be disjoint, got {sorted(overlap)}"
            )
        if r_conditioned_local_tail_window_sizes is None:
            r_conditioned_local_tail_window_sizes = {"res2": 3, "res3": 5}
        self.r_conditioned_local_tail_window_sizes = dict(
            r_conditioned_local_tail_window_sizes
        )
        for scale in self.r_conditioned_local_tail_scales:
            window_size = int(
                self.r_conditioned_local_tail_window_sizes.get(scale, 3)
            )
            if window_size < 3 or window_size % 2 != 1:
                raise ValueError(
                    "Every R-conditioned local tail window must be an odd integer >= 3"
                )
        if tail_dual_memory_context_enabled and (
            attention_direction != "anchor_to_tail" or tail_aggregation != "attention"
        ):
            raise ValueError(
                "tail_dual_memory_context requires anchor_to_tail attention "
                "and tail_aggregation='attention'"
            )
        if tail_dual_memory_context_scales is None:
            tail_dual_memory_context_scales = (
                ("res3",) if tail_dual_memory_context_enabled else ()
            )
        unknown_tail_dual_memory_scales = set(tail_dual_memory_context_scales) - set(
            _SCALES[:num_scales]
        )
        if unknown_tail_dual_memory_scales:
            raise ValueError(
                "Unknown/unavailable tail_dual_memory_context_scales: "
                f"{sorted(unknown_tail_dual_memory_scales)}"
            )
        self.tail_dual_memory_context_scales = frozenset(
            tail_dual_memory_context_scales if tail_dual_memory_context_enabled else ()
        )
        self.tail_anchor_memory_mode = tail_anchor_memory_mode
        self.tail_anchor_local_window_size = int(tail_anchor_local_window_size)
        self.tail_anchor_local_scales = frozenset(tail_anchor_local_scales)
        self.tail_anchor_local_query_chunk_size = int(tail_anchor_local_query_chunk_size)
        self.tail_anchor_local_checkpoint = bool(tail_anchor_local_checkpoint)
        if tdom_memory_token_scales is None:
            tdom_memory_token_scales = _SCALES[:num_scales] if tdom_memory_token_enabled else ()
        unknown_tdom_memory_scales = set(tdom_memory_token_scales) - set(
            _SCALES[:num_scales]
        )
        if unknown_tdom_memory_scales:
            raise ValueError(
                "Unknown/unavailable tdom_memory_token_scales: "
                f"{sorted(unknown_tdom_memory_scales)}"
            )
        self.tdom_memory_token_enabled = bool(tdom_memory_token_enabled)
        self.tdom_memory_token_readout = str(tdom_memory_token_readout)
        self.tdom_memory_token_scales = frozenset(
            tdom_memory_token_scales if tdom_memory_token_enabled else ()
        )
        if tdom_in_anchor_memory_scales is None:
            tdom_in_anchor_memory_scales = (
                _SCALES[:num_scales] if tdom_in_anchor_memory else ()
            )
        unknown_tdom_in_anchor_scales = set(tdom_in_anchor_memory_scales) - set(
            _SCALES[:num_scales]
        )
        if unknown_tdom_in_anchor_scales:
            raise ValueError(
                "Unknown/unavailable tdom_in_anchor_memory_scales: "
                f"{sorted(unknown_tdom_in_anchor_scales)}"
            )
        self.tdom_in_anchor_memory = bool(tdom_in_anchor_memory)
        self.tdom_in_anchor_memory_scales = frozenset(
            tdom_in_anchor_memory_scales if tdom_in_anchor_memory else ()
        )
        self.memory_output_mode = memory_output_mode
        # Keep the historical behaviour by default.  When disabled, a zero
        # feature at a pixel of a real tail view remains a valid attention
        # candidate.  Non-existent/padded view slots are still invalid.  This
        # switch deliberately affects only the tail reader: anchor zero masks
        # and therefore the learned R/support semantics remain unchanged.
        self.mask_zero_tail_features = bool(mask_zero_tail_features)
        self._local_tail_reader_scales = frozenset(
            self.tail_anchor_local_scales
            if self.tail_anchor_memory_mode == "local_token_set"
            and self.attention_direction == "tail_to_anchor"
            else ()
        )
        # The null key participates in R-query attention but has an exactly
        # zero value.  It therefore represents the explicit option of not
        # applying a tail update, rather than forcing attention to select the
        # least-bad projected image feature.
        self.use_null_tail = bool(use_null_tail)
        self.anchor_blocks = nn.ModuleList([
            nn.ModuleList([
                _ViewAxisTransformerBlock(embed_dim, num_heads, dropout, ffn_ratio)
                for _ in range(anchor_depth)
            ])
            for _ in range(num_scales)
        ])
        self.anchor_spatial_blocks = nn.ModuleList([
            nn.ModuleList([
                _CrossViewLocalAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    window_size=self.anchor_spatial_window_size,
                    dropout=dropout,
                    ffn_ratio=ffn_ratio,
                    residual_init=anchor_spatial_residual_init,
                )
                for _ in range(anchor_spatial_depth)
            ]) if (
                self.anchor_spatial_window_size > 1 and scale in self.anchor_spatial_scales
            ) else nn.ModuleList()
            for scale in _SCALES[:num_scales]
        ])
        # One independent resolution-preserving Swin refiner per selected
        # FPN scale.  Non-selected scales use Identity so the baseline has no
        # extra parameters or altered computation.
        self.swin_r_refiners = nn.ModuleList([
            SwinRRefiner(
                embed_dim=embed_dim,
                window_size=swin_r_window_size,
                num_heads=swin_r_num_heads,
                depth=swin_r_depth,
                mlp_ratio=swin_r_mlp_ratio,
                dropout=swin_r_dropout,
                alpha_init=swin_r_alpha_init,
            ) if scale in self.swin_r_scales else nn.Identity()
            for scale in _SCALES[:num_scales]
        ])
        self.multi_latent_refiners = nn.ModuleList([
            MultiLatentSwinR(
                embed_dim=embed_dim,
                num_latents=multi_latent_num,
                num_heads=swin_r_num_heads,
                spatial_window_size=multi_latent_spatial_window_size,
                spatial_depth=multi_latent_spatial_depth,
                latent_depth=multi_latent_depth,
                mlp_ratio=swin_r_mlp_ratio,
                dropout=swin_r_dropout,
                beta_init=multi_latent_beta_init,
                alpha_init=multi_latent_alpha_init,
            ) if scale in self.multi_latent_scales else nn.Identity()
            for scale in _SCALES[:num_scales]
        ])
        self.single_r_swin_set_refiners = nn.ModuleList([
            SingleRSwinSet(
                embed_dim=embed_dim,
                num_heads=num_heads,
                spatial_window_size=single_r_spatial_window_size,
                spatial_depth=single_r_spatial_depth,
                view_depth=single_r_view_depth,
                mlp_ratio=swin_r_mlp_ratio,
                dropout=swin_r_dropout,
                spatial_alpha_init=single_r_spatial_alpha_init,
                beta_init=single_r_beta_init,
            ) if scale in self.single_r_swin_set_scales else nn.Identity()
            for scale in _SCALES[:num_scales]
        ])
        # A single lightweight residual CNN is shared by all tail views at a
        # scale.  Unlike R construction, this path never performs tail-to-tail
        # attention: it only supplies local ground-plane context to each tail.
        self.tail_local_context_refiners = nn.ModuleList([
            TailLocalContext(
                embed_dim=embed_dim,
                kernel_size=tail_local_context_kernel_size,
                bottleneck_ratio=tail_local_context_bottleneck_ratio,
                dropout=tail_local_context_dropout,
                alpha_init=tail_local_context_alpha_init,
            ) if scale in self.tail_local_context_scales else nn.Identity()
            for scale in _SCALES[:num_scales]
        ])
        # At selected scales, R(x,y) directly queries a local neighbourhood
        # from every tail.  This replaces both TailLocal preprocessing and the
        # ordinary same-pixel tail MHA at those scales.
        self.r_conditioned_local_tail_blocks = nn.ModuleList([
            RConditionedLocalTailAttention(
                embed_dim=embed_dim,
                attention_dim=r_conditioned_local_tail_attention_dim,
                num_heads=r_conditioned_local_tail_num_heads,
                window_size=int(
                    self.r_conditioned_local_tail_window_sizes.get(scale, 3)
                ),
                ffn_ratio=r_conditioned_local_tail_ffn_ratio,
                dropout=r_conditioned_local_tail_dropout,
                eps=eps,
            ) if scale in self.r_conditioned_local_tail_scales else nn.Identity()
            for scale in _SCALES[:num_scales]
        ])
        self.tail_dual_memory_context_refiners = nn.ModuleList([
            DualMemoryTailCalibrator(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=tail_dual_memory_context_ffn_ratio,
                dropout=dropout,
                alpha_init=tail_dual_memory_context_alpha_init,
            ) if scale in self.tail_dual_memory_context_scales else nn.Identity()
            for scale in _SCALES[:num_scales]
        ])
        # The anchor memory and TDOM feature remain separate queries until both
        # have read the same tail set.  The legacy MLP readout is preserved by
        # default; the optional three-way gate explicitly selects anchor,
        # TDOM, or no tail residual at each pixel.
        self.tdom_memory_token_readouts = nn.ModuleList([
            (
                ThreeWayGatedTailReadout(
                    embed_dim, hidden_dim=tdom_memory_gate_hidden_dim
                )
                if self.tdom_memory_token_readout == "three_way_gate"
                else TwoTokenTailReadout(embed_dim, tdom_memory_token_hidden_ratio)
            )
            if scale in self.tdom_memory_token_scales else nn.Identity()
            for scale in _SCALES[:num_scales]
        ])
        # TDOM and projected-image features share the same 256-D neck space.
        # Register this only when the branch is enabled: a trainable parameter
        # that is never visited in forward would break ordinary DDP.
        if self.tdom_in_anchor_memory:
            self.tdom_in_anchor_memory_alpha = nn.Parameter(torch.full(
                (num_scales,), float(tdom_in_anchor_memory_alpha_init)
            ))
        else:
            self.register_parameter("tdom_in_anchor_memory_alpha", None)
        self.tail_anchor_local_blocks = nn.ModuleList([
            _TailQueryLocalAnchorMemory(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=self.tail_anchor_local_window_size,
                dropout=dropout,
                ffn_ratio=ffn_ratio,
                residual_init=tail_anchor_local_residual_init,
            ) if scale in self._local_tail_reader_scales else nn.Identity()
            for scale in _SCALES[:num_scales]
        ])
        special_tail_reader_scales = (
            self._local_tail_reader_scales
            | self.r_conditioned_local_tail_scales
        )
        if self.tail_aggregation == "direct_mean":
            # Direct mean has no Q/K/V projections or trainable tail-side
            # normalizers. Identity modules also prevent unused DDP parameters.
            special_tail_reader_scales = frozenset(_SCALES[:num_scales])
        self.tail_norm_q = nn.ModuleList([
            nn.Identity() if scale in special_tail_reader_scales else nn.LayerNorm(embed_dim)
            for scale in _SCALES[:num_scales]
        ])
        self.tail_norm_kv = nn.ModuleList([
            nn.Identity() if scale in special_tail_reader_scales else nn.LayerNorm(embed_dim)
            for scale in _SCALES[:num_scales]
        ])
        # The sparse tailwise path uses only the two normalizers above plus
        # its own candidate/score MLPs.  Do not register unused attention
        # modules in that path: DDP correctly treats such parameters as an
        # error when they never contribute to the training loss.
        needs_attention_tail_modules = (
            self.tail_aggregation in {"attention", "global_tail_topk"}
            or self.attention_direction == "tail_to_anchor"
        )
        if needs_attention_tail_modules:
            self.tail_attn = nn.ModuleList([
                nn.Identity() if scale in special_tail_reader_scales else
                nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
                for scale in _SCALES[:num_scales]
            ])
            self.tail_norm_ffn = nn.ModuleList([
                nn.Identity() if scale in special_tail_reader_scales else nn.LayerNorm(embed_dim)
                for scale in _SCALES[:num_scales]
            ])
            self.tail_ffn = nn.ModuleList([
                nn.Identity() if scale in special_tail_reader_scales else nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * ffn_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(embed_dim * ffn_ratio), embed_dim),
                    nn.Dropout(dropout),
                )
                for scale in _SCALES[:num_scales]
            ])
        else:
            self.tail_attn = nn.ModuleList()
            self.tail_norm_ffn = nn.ModuleList()
            self.tail_ffn = nn.ModuleList()
        if self.tail_value_mode == "raw_weighted":
            # Keep keys for strict loading of old checkpoints, but exclude
            # bypassed modules from training/DDP. Packed Q/K/V is one parameter;
            # only its Q/K slices receive gradients in this mode.
            self.tail_norm_ffn.requires_grad_(False)
            self.tail_ffn.requires_grad_(False)
            for attention in self.tail_attn:
                attention.out_proj.requires_grad_(False)
        if self.memory_output_mode == "r_plus_attention":
            # Retain checkpoint keys but bypass and exclude these modules from
            # optimization/DDP. A still trains Q/K/V and output projections.
            self.tail_norm_ffn.requires_grad_(False)
            self.tail_ffn.requires_grad_(False)
        if self.use_null_tail:
            self.null_tail_keys = nn.Parameter(torch.empty(num_scales, embed_dim))
            nn.init.normal_(self.null_tail_keys, mean=0.0, std=0.02)
            # Start slightly less attractive than a valid image tail.  This
            # avoids an early all-null collapse while leaving the prior fully
            # learnable from the segmentation loss.
            self.null_tail_logit = nn.Parameter(
                torch.full((num_scales,), float(null_tail_logit_init))
            )
        else:
            self.register_parameter("null_tail_keys", None)
            self.register_parameter("null_tail_logit", None)
        if self.tail_aggregation == "tailwise_topk":
            # Each projected image proposes an independent residual update and
            # a confidence score.  A subsequent Top-K aggregate is sparse over
            # views, unlike regular attention where every valid tail receives
            # non-zero softmax mass.
            self.tailwise_update = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim * 3, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(num_scales)
            ])
            hidden_score_dim = max(embed_dim // 4, 16)
            self.tailwise_score = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim * 3, hidden_score_dim),
                    nn.GELU(),
                    nn.Linear(hidden_score_dim, 1),
                )
                for _ in range(num_scales)
            ])
        else:
            self.tailwise_update = None
            self.tailwise_score = None
        if self.tail_aggregation == "global_tail_topk":
            # One scalar per view and patch.  Scores are computed independently
            # for every tail from its relation to R at one reference scale, so
            # scores remain comparable across chunks during all-tail inference.
            hidden_score_dim = max(embed_dim // 2, 32)
            score_input_dim = embed_dim * 3 + 1
            self.tail_global_score = nn.Sequential(
                nn.LayerNorm(score_input_dim),
                nn.Linear(score_input_dim, hidden_score_dim),
                nn.GELU(),
                nn.Linear(hidden_score_dim, 1),
            )
            # Begin as an unbiased selector.  Candidate identities are already
            # randomised by the dataset, while segmentation gradients can
            # immediately differentiate the selected scores.
            nn.init.zeros_(self.tail_global_score[-1].weight)
            nn.init.zeros_(self.tail_global_score[-1].bias)
        else:
            self.tail_global_score = None
        self.alpha = nn.Parameter(torch.full((num_scales,), float(alpha_init)))
        if final_fusion not in {"add", "concat_project"}:
            raise ValueError("final_fusion must be add or concat_project")
        self.final_fusion = final_fusion
        if final_fusion == "concat_project":
            self.final_projections = nn.ModuleList([
                nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=True)
                for _ in range(num_scales)
            ])
            # Alpha is absorbed into the image-side projection at init. Keep
            # its checkpoint key, but do not leave an unused trainable scalar.
            self.alpha.requires_grad_(False)
            self.reset_final_projections_to_add()
        if self.memory_output_mode == "r_plus_attention":
            self.register_buffer("tail_beta", torch.ones(num_scales))
        elif self.tail_aggregation == "direct_mean":
            # This ablation is deliberately an unweighted R + TailMean sum.
            # Keep beta in the state dict/aux interface, but make it a fixed
            # buffer so training cannot silently suppress the mean branch.
            self.register_buffer(
                "tail_beta", torch.full((num_scales,), float(tail_beta_init))
            )
        else:
            self.tail_beta = nn.Parameter(
                torch.full((num_scales,), float(tail_beta_init))
            )

    @torch.no_grad()
    def reset_final_projections_to_add(self):
        """Call AFTER loading the additive checkpoint to use its actual alpha."""
        for sid, projection in enumerate(self.final_projections):
            channels = projection.out_channels
            identity = torch.eye(channels, device=projection.weight.device,
                                 dtype=projection.weight.dtype)
            projection.weight.zero_()
            projection.weight[:, :channels, 0, 0].copy_(identity)
            projection.weight[:, channels:, 0, 0].copy_(
                identity * self.alpha[sid].to(identity))
            projection.bias.zero_()

    def _fuse_tdom_memory(self, tdom, memory, scale_id):
        if self.final_fusion == "concat_project":
            return self.final_projections[scale_id](torch.cat([tdom, memory], dim=1))
        alpha = self.alpha[scale_id].to(dtype=tdom.dtype, device=tdom.device)
        return tdom + alpha * memory

    def _effective_tail_beta(self, scale_id, reference):
        if self.memory_output_mode == "r_plus_attention":
            return reference.new_ones(())
        return self.tail_beta[scale_id].to(dtype=reference.dtype, device=reference.device)

    def _merge_anchor_tail(self, anchor, tail, beta, *, conditioned_path=False):
        if self.memory_output_mode == "r_plus_attention":
            return anchor + tail  # X=R+A: no beta multiplication, no FFN.
        if conditioned_path and self.memory_output_mode == "tail_conditioned":
            return beta * tail
        return anchor + beta * tail

    @staticmethod
    def _stack_views(
        features,
        scale: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not features:
            return None
        values = []
        for view in features:
            if scale not in view:
                raise KeyError(f"View feature is missing {scale!r}")
            value = view[scale]
            if value.ndim == 3:
                value = value.unsqueeze(0)
            values.append(value.to(device=device, dtype=dtype))
        return torch.stack(values, dim=1)  # [B, K, C, H, W]

    @staticmethod
    def _stack_selected_views(
        features,
        scale: str,
        indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Materialise only batch-specific selected views on the target GPU."""
        if not features:
            return None
        selected_batches = []
        for batch_index, view_indices in enumerate(indices.detach().cpu().tolist()):
            selected = []
            for view_index in view_indices:
                view = features[view_index]
                if scale not in view:
                    raise KeyError(f"View feature is missing {scale!r}")
                value = view[scale]
                if value.ndim == 3:
                    if indices.shape[0] != 1:
                        raise ValueError(
                            "Unbatched tail feature is only valid for batch_size=1"
                        )
                    selected.append(value)
                else:
                    selected.append(value[batch_index])
            selected_batches.append(torch.stack(selected, dim=0))
        return torch.stack(selected_batches, dim=0).to(
            device=device, dtype=dtype, non_blocking=True
        )

    def _valid_pixels(
        self,
        views: torch.Tensor,
        view_exist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # [B,K,H,W]; zeros are invalid projected feature, even for an existing camera.
        valid = views.abs().sum(dim=2) > self.eps
        if view_exist is not None:
            exists = view_exist.to(device=views.device, dtype=torch.bool)
            if exists.shape != views.shape[:2]:
                raise ValueError(
                    f"view_exist must be {tuple(views.shape[:2])}, got {tuple(exists.shape)}"
                )
            valid = valid & exists[:, :, None, None]
        return valid

    def _tail_attention_valid_pixels(
        self,
        views: torch.Tensor,
        view_exist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return the mask used when tails enter cross-view attention.

        With ``mask_zero_tail_features=False``, an all-channel zero vector
        from an existing projected view is exposed to attention instead of
        being discarded.  Padding is never exposed.  TailLocal still uses the
        ordinary content mask, so an input zero stays a zero token rather than
        being synthesized from neighbouring pixels before attention.
        """
        if self.mask_zero_tail_features:
            return self._valid_pixels(views, view_exist)

        valid = torch.ones(
            views.shape[0], views.shape[1], views.shape[-2], views.shape[-1],
            dtype=torch.bool, device=views.device,
        )
        if view_exist is not None:
            exists = view_exist.to(device=views.device, dtype=torch.bool)
            if exists.shape != views.shape[:2]:
                raise ValueError(
                    f"view_exist must be {tuple(views.shape[:2])}, got {tuple(exists.shape)}"
                )
            valid = valid & exists[:, :, None, None]
        return valid

    def _contextualize_tail_views(
        self,
        tail_views: Optional[torch.Tensor],
        tail_exist: Optional[torch.Tensor],
        scale_id: int,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Apply the shared lightweight local block independently per tail.

        Chunking limits temporary convolution activations for Q=16 while the
        returned tensor preserves the original ``[B,Q,C,H,W]`` layout and all
        tail identities.  No cross-view operation is introduced here.
        """
        if tail_views is None:
            zero = self.alpha.new_zeros(())
            return None, zero, zero
        refiner = self.tail_local_context_refiners[scale_id]
        if isinstance(refiner, nn.Identity):
            zero = tail_views.new_zeros(())
            return tail_views, zero, zero

        batch, tail_count, channels, height, width = tail_views.shape
        valid = self._valid_pixels(tail_views, tail_exist)
        outputs = []
        alphas = []
        ratios = []
        for start in range(0, tail_count, self.tail_local_context_chunk_size):
            end = min(start + self.tail_local_context_chunk_size, tail_count)
            chunk_size = end - start
            flat_tail = tail_views[:, start:end].reshape(
                batch * chunk_size, channels, height, width
            )
            flat_valid = valid[:, start:end].reshape(
                batch * chunk_size, height, width
            )
            contextualized, alpha, ratio = refiner(flat_tail, flat_valid)
            outputs.append(contextualized.reshape(
                batch, chunk_size, channels, height, width
            ))
            alphas.append(alpha)
            ratios.append(ratio)
        return torch.cat(outputs, dim=1), torch.stack(alphas).mean(), torch.stack(ratios).mean()

    def _condition_tails_on_dual_memory(
        self,
        tail_views: Optional[torch.Tensor],
        tail_exist: Optional[torch.Tensor],
        anchor_memory: torch.Tensor,
        anchor_support: torch.Tensor,
        tdom_memory: torch.Tensor,
        scale_id: int,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Let each tail read the paired R/TDOM memory before R reads tails."""
        if tail_views is None:
            zero = anchor_memory.new_zeros(())
            return None, zero, zero
        refiner = self.tail_dual_memory_context_refiners[scale_id]
        if isinstance(refiner, nn.Identity):
            zero = tail_views.new_zeros(())
            return tail_views, zero, zero
        # Preserve raw zero locations through the optional pre-attention
        # calibrator.  They can still be exposed as zero candidates by the
        # downstream tail reader.
        tail_valid = self._valid_pixels(tail_views, tail_exist)
        anchor_valid = anchor_support[:, 0] > 0
        tdom_valid = self._valid_pixels(
            tdom_memory.unsqueeze(1), view_exist=None
        )[:, 0]
        return refiner(
            tail_views,
            tail_valid,
            anchor_memory,
            anchor_valid,
            tdom_memory,
            tdom_valid,
        )

    def _global_tail_scores(
        self,
        tail_views: Optional[torch.Tensor],
        tail_exist: Optional[torch.Tensor],
        anchor_memory: torch.Tensor,
        anchor_support: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return one independently calibrated score per patch and tail.

        The scorer uses masked spatial summaries of R, the tail, their
        absolute disagreement, and the usable-overlap ratio.  It never
        normalizes across the candidate axis, which keeps scores comparable
        when all-tail inference is split into arbitrary-sized chunks.
        """
        if tail_views is None:
            return None
        if self.tail_global_score is None:
            raise RuntimeError("global tail scores requested without scorer")
        b, q, c, h, w = tail_views.shape
        if anchor_memory.shape != (b, c, h, w):
            raise ValueError("Global tail scorer requires tail/R shape match")

        tail_valid = self._tail_attention_valid_pixels(tail_views, tail_exist)
        anchor_valid = anchor_support[:, 0] > 0
        usable = tail_valid & anchor_valid[:, None]
        # Pool in FP32 even when feature tensors are FP16.  In particular, the
        # absolute-difference sum over a 64x64 res3 map can overflow FP16.
        weights = usable[:, :, None].float()
        count = weights.sum(dim=(-2, -1)).clamp_min(1.0)
        expanded_r = anchor_memory[:, None].float()
        tail_float = tail_views.float()
        r_pool = (expanded_r * weights).sum(dim=(-2, -1)) / count
        tail_pool = (tail_float * weights).sum(dim=(-2, -1)) / count
        diff_pool = (
            (tail_float - expanded_r).abs() * weights
        ).sum(dim=(-2, -1)) / count
        overlap_ratio = usable.float().mean(dim=(-2, -1), keepdim=False)
        descriptor = torch.cat(
            (r_pool, tail_pool, diff_pool, overlap_ratio[..., None].to(r_pool.dtype)),
            dim=-1,
        )
        raw_scores = self.tail_global_score(descriptor).squeeze(-1)
        # A bounded monotonic map preserves ranking but prevents a learned
        # global prior from driving attention logits to +/-inf.
        scores = 5.0 * torch.tanh(raw_scores / 5.0)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=5.0, neginf=-5.0)
        usable_view = usable.flatten(2).any(dim=-1)
        return scores.masked_fill(~usable_view, -1.0e4)

    def _active_tail_topk(self, candidate_count: int) -> int:
        """Choose K for this forward; validation/prediction use tail_topk."""
        upper = min(self.tail_topk, int(candidate_count))
        if upper < 1:
            return 0
        if self.training and self.tail_topk_random_train:
            lower = min(self.tail_topk_min, upper)
            return int(torch.randint(lower, upper + 1, ()).item())
        return min(self.tail_topk_eval, upper)

    @staticmethod
    def _gather_tail_axis(
        values: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Batch-specific gather along the view dimension."""
        if values.ndim < 2 or values.shape[:2] != (
            indices.shape[0], values.shape[1]
        ):
            raise ValueError("Tail gather expects [B,Q,...] values and [B,K] indices")
        gather_index = indices
        for _ in range(values.ndim - 2):
            gather_index = gather_index.unsqueeze(-1)
        gather_index = gather_index.expand(
            *indices.shape, *values.shape[2:]
        )
        return torch.gather(values, dim=1, index=gather_index)

    @staticmethod
    def _to_view_tokens(views: torch.Tensor, valid: torch.Tensor):
        # [B,K,C,H,W] -> [B*H*W,K,C], preserving independent ground pixels.
        b, k, c, h, w = views.shape
        tokens = views.permute(0, 3, 4, 1, 2).reshape(b * h * w, k, c)
        valid_tokens = valid.permute(0, 2, 3, 1).reshape(b * h * w, k)
        return tokens, valid_tokens

    @staticmethod
    def _from_memory_tokens(tokens: torch.Tensor, batch: int, channels: int,
                            height: int, width: int) -> torch.Tensor:
        return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

    @staticmethod
    def _from_view_tokens(
        tokens: torch.Tensor, batch: int, channels: int, height: int, width: int
    ) -> torch.Tensor:
        """``[B*H*W,K,C]`` back to the unpooled ``[B,K,C,H,W]`` memory set."""
        k = tokens.shape[1]
        return tokens.reshape(batch, height, width, k, channels).permute(0, 3, 4, 1, 2)

    @staticmethod
    def _from_view_valid_tokens(
        valid: torch.Tensor, batch: int, height: int, width: int
    ) -> torch.Tensor:
        k = valid.shape[1]
        return valid.reshape(batch, height, width, k).permute(0, 3, 1, 2)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = valid.unsqueeze(-1).to(dtype=tokens.dtype)
        count = weights.sum(dim=1)
        mean = (tokens * weights).sum(dim=1) / count.clamp_min(1.0)
        mean = torch.where(count > 0, mean, torch.zeros_like(mean))
        return mean, count

    def _inject_tdom_into_anchor_memory(
        self,
        memory: torch.Tensor,
        reference: torch.Tensor,
        scale_id: int,
    ) -> torch.Tensor:
        """Construct one co-adapted R without altering anchor support.

        ``reference`` is the TDOM neck feature at the same FPN scale.  It is
        intentionally injected after the anchor-set refiner: the latter still
        learns pure multi-view consensus, while the resulting single R can use
        TDOM evidence in both tail attention and the residual output path.
        """
        if _SCALES[scale_id] not in self.tdom_in_anchor_memory_scales:
            return memory
        valid_tdom = self._valid_pixels(
            reference.unsqueeze(1), view_exist=None
        ).to(dtype=reference.dtype)
        alpha = self.tdom_in_anchor_memory_alpha[scale_id].to(
            device=reference.device, dtype=reference.dtype
        )
        return memory + alpha * reference * valid_tdom

    def _anchor_memory(
        self,
        views: Optional[torch.Tensor],
        view_exist: Optional[torch.Tensor],
        scale_id: int,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return memory/support in map form and contextualized anchor tokens."""
        if views is None:
            b, _, h, w = reference.shape
            empty_memory = torch.zeros_like(reference)
            empty_support = torch.zeros((b, 1, h, w), device=reference.device, dtype=reference.dtype)
            zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
            empty_memory = self._inject_tdom_into_anchor_memory(
                empty_memory, reference, scale_id
            )
            return empty_memory, empty_support, None, None, zero, zero, zero, zero, zero

        b, k, c, h, w = views.shape
        if (b, c, h, w) != (reference.shape[0], reference.shape[1], *reference.shape[-2:]):
            raise ValueError("Anchor feature shape must match the TDOM feature at each scale")
        valid = self._valid_pixels(views, view_exist)
        contextualized_views = views
        # This happens before per-pixel view-axis attention.  The local block
        # keeps anchors as an unordered set and therefore does not give a
        # privileged role to the arbitrary first K-medoid.
        for block in self.anchor_spatial_blocks[scale_id]:
            contextualized_views = block(contextualized_views, valid)
        tokens, token_valid = self._to_view_tokens(contextualized_views, valid)
        single_r = self.single_r_swin_set_refiners[scale_id]
        if isinstance(single_r, SingleRSwinSet):
            (
                memory, support, set_views, set_valid,
                swin_alpha, delta_ratio, _pool_weights,
            ) = single_r(contextualized_views, valid)
            set_tokens, set_token_valid = self._to_view_tokens(set_views, set_valid)
            memory = self._inject_tdom_into_anchor_memory(memory, reference, scale_id)
            zero = torch.zeros((), device=memory.device, dtype=memory.dtype)
            return (
                memory, support, set_tokens, set_token_valid,
                swin_alpha, delta_ratio, zero, zero, zero,
            )
        multi_latent = self.multi_latent_refiners[scale_id]
        if isinstance(multi_latent, MultiLatentSwinR):
            (
                memory, support, latent_maps, latent_valid, _weights,
                swin_alpha, delta_ratio, latent_stats, weight_mean,
            ) = multi_latent(contextualized_views, valid)
            latent_tokens, latent_token_valid = self._to_view_tokens(
                latent_maps, latent_valid.bool()
            )
            memory = self._inject_tdom_into_anchor_memory(memory, reference, scale_id)
            return (
                memory, support, latent_tokens, latent_token_valid,
                swin_alpha, delta_ratio, latent_stats[0], latent_stats[1], weight_mean,
            )
        contextualized = tokens
        for block in self.anchor_blocks[scale_id]:
            contextualized = block(contextualized, token_valid)
        mean, count = self._masked_mean(contextualized, token_valid)
        memory = self._from_memory_tokens(mean, b, c, h, w)
        support = self._from_memory_tokens(count, b, 1, h, w) / max(k, 1)
        swin_r = self.swin_r_refiners[scale_id]
        if isinstance(swin_r, SwinRRefiner):
            memory, swin_alpha, delta_ratio = swin_r(
                memory, (support[:, 0] > 0)
            )
        else:
            swin_alpha = torch.zeros((), device=memory.device, dtype=memory.dtype)
            delta_ratio = torch.zeros((), device=memory.device, dtype=memory.dtype)
        memory = self._inject_tdom_into_anchor_memory(memory, reference, scale_id)
        zero = torch.zeros((), device=memory.device, dtype=memory.dtype)
        return memory, support, contextualized, token_valid, swin_alpha, delta_ratio, zero, zero, zero

    def _tail_memory(
        self,
        tail_views: Optional[torch.Tensor],
        tail_exist: Optional[torch.Tensor],
        anchor_tokens: Optional[torch.Tensor],
        anchor_valid: Optional[torch.Tensor],
        scale_id: int,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tail_sum, tail_count, tail_slots = self._tail_memory_sum(
            tail_views, tail_exist, anchor_tokens, anchor_valid, scale_id, reference
        )
        if tail_slots == 0:
            b, _, h, w = reference.shape
            empty_memory = torch.zeros_like(reference)
            empty_support = torch.zeros((b, 1, h, w), device=reference.device, dtype=reference.dtype)
            return empty_memory, empty_support

        memory = tail_sum / tail_count.clamp_min(1.0)
        memory = torch.where(tail_count > 0, memory, torch.zeros_like(memory))
        support = tail_count / float(tail_slots)
        return memory, support

    def _direct_tail_mean(
        self,
        tail_views: Optional[torch.Tensor],
        tail_exist: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use raw/pre-averaged tail features without attention or projection."""
        b, _, h, w = reference.shape
        empty = torch.zeros_like(reference)
        empty_support = torch.zeros(
            (b, 1, h, w), device=reference.device, dtype=reference.dtype
        )
        if tail_views is None:
            return empty, empty_support
        bt, q, c, ht, wt = tail_views.shape
        if (bt, c, ht, wt) != (b, reference.shape[1], h, w):
            raise ValueError("Tail feature shape must match the TDOM feature at each scale")
        valid = self._valid_pixels(tail_views, tail_exist)
        weights = valid[:, :, None].to(dtype=tail_views.dtype)
        count = weights.sum(dim=1)
        memory = (tail_views * weights).sum(dim=1) / count.clamp_min(1.0)
        memory = torch.where(count > 0, memory, torch.zeros_like(memory))
        support = count / float(max(q, 1))
        return memory, support

    def _tail_memory_sum(
        self,
        tail_views: Optional[torch.Tensor],
        tail_exist: Optional[torch.Tensor],
        anchor_tokens: Optional[torch.Tensor],
        anchor_valid: Optional[torch.Tensor],
        scale_id: int,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return calibrated tail sum and valid-tail count in map form.

        Keeping sums rather than a chunk mean makes all-tail inference exactly
        invariant to the number of tail chunks.
        """
        if tail_views is None or anchor_tokens is None:
            b, _, h, w = reference.shape
            empty_memory = torch.zeros_like(reference)
            empty_count = torch.zeros((b, 1, h, w), device=reference.device, dtype=reference.dtype)
            return empty_memory, empty_count, 0

        b, q, c, h, w = tail_views.shape
        if (b, c, h, w) != (reference.shape[0], reference.shape[1], *reference.shape[-2:]):
            raise ValueError("Tail feature shape must match the TDOM feature at each scale")
        tail_valid_map = self._tail_attention_valid_pixels(tail_views, tail_exist)
        local_reader = self.tail_anchor_local_blocks[scale_id]
        if isinstance(local_reader, _TailQueryLocalAnchorMemory):
            anchor_maps = self._from_view_tokens(anchor_tokens, b, c, h, w)
            anchor_valid_map = self._from_view_valid_tokens(anchor_valid, b, h, w)
            # A dense local read at Q=32 retains attention intermediates for
            # every tail and every offset during backward.  Process query
            # views in small independent groups and checkpoint their local
            # attention; the result is mathematically the same tail sum/count
            # while being viable on the available GPUs.
            memory_sum = torch.zeros_like(reference)
            count_map = torch.zeros(
                (b, 1, h, w), device=reference.device, dtype=reference.dtype
            )
            for start in range(0, q, self.tail_anchor_local_query_chunk_size):
                end = min(start + self.tail_anchor_local_query_chunk_size, q)
                chunk_views = tail_views[:, start:end]
                chunk_valid_map = tail_valid_map[:, start:end]
                if self.training and self.tail_anchor_local_checkpoint:
                    calibrated_maps, usable_map = checkpoint(
                        local_reader,
                        chunk_views,
                        chunk_valid_map,
                        anchor_maps,
                        anchor_valid_map,
                        use_reentrant=False,
                    )
                else:
                    calibrated_maps, usable_map = local_reader(
                        chunk_views, chunk_valid_map, anchor_maps, anchor_valid_map
                    )
                calibrated, usable = self._to_view_tokens(calibrated_maps, usable_map)
                mean, count = self._masked_mean(calibrated, usable)
                memory_sum = memory_sum + self._from_memory_tokens(mean * count, b, c, h, w)
                count_map = count_map + self._from_memory_tokens(count, b, 1, h, w)
            return memory_sum, count_map, q
        else:
            tail_tokens, tail_valid = self._to_view_tokens(tail_views, tail_valid_map)
            key_padding_mask, empty_anchor = _safe_key_padding_mask(anchor_valid)
            attended, _ = self.tail_attn[scale_id](
                self.tail_norm_q[scale_id](tail_tokens),
                self.tail_norm_kv[scale_id](anchor_tokens),
                self.tail_norm_kv[scale_id](anchor_tokens),
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            calibrated = tail_tokens + attended
            calibrated = calibrated + self.tail_ffn[scale_id](self.tail_norm_ffn[scale_id](calibrated))
            calibrated = calibrated.masked_fill(~tail_valid.unsqueeze(-1), 0.0)
            if empty_anchor.any():
                # A tail without an anchor representation must not create memory.
                calibrated[empty_anchor] = 0.0
                tail_valid = tail_valid.clone()
                tail_valid[empty_anchor] = False
        mean, count = self._masked_mean(calibrated, tail_valid)
        memory_sum = self._from_memory_tokens(mean * count, b, c, h, w)
        count_map = self._from_memory_tokens(count, b, 1, h, w)
        return memory_sum, count_map, q

    def _compose_anchor_tail_delta(
        self, query: torch.Tensor, attended: torch.Tensor, scale_id: int
    ) -> torch.Tensor:
        """Keep FFN(R) stable while scaling only actual tail-dependent content.

        Eligibility masking remains at the caller, including the empty-tail
        case. Shared FFN weights are trained by both terms; no detach or new
        trainable parameters are introduced.
        """
        if self.memory_output_mode == "r_plus_attention":
            return attended
        full = attended + self.tail_ffn[scale_id](
            self.tail_norm_ffn[scale_id](query + attended)
        )
        if self.tail_content_gamma == 1 or _SCALES[scale_id] not in self.tail_content_gamma_scales:
            return full
        stable = self.tail_ffn[scale_id](self.tail_norm_ffn[scale_id](query))
        return stable + self.tail_content_gamma * (full - stable)

    def _anchor_query_tail_update(
        self,
        tail_views: Optional[torch.Tensor],
        tail_exist: Optional[torch.Tensor],
        anchor_memory: torch.Tensor,
        anchor_support: torch.Tensor,
        scale_id: int,
        reference: torch.Tensor,
        tail_score_bias: Optional[torch.Tensor] = None,
        raw_tail_views: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use compact anchor memory as Q and valid tail views as K/V."""
        b, c, h, w = reference.shape
        empty = torch.zeros_like(reference)
        empty_support = torch.zeros(
            (b, 1, h, w), device=reference.device, dtype=reference.dtype
        )
        if tail_views is None:
            return empty, empty_support

        bt, q, ct, ht, wt = tail_views.shape
        if (bt, ct, ht, wt) != (b, c, h, w):
            raise ValueError("Tail feature shape must match the TDOM feature at each scale")

        tail_valid_map = self._tail_attention_valid_pixels(tail_views, tail_exist)
        local_reader = self.r_conditioned_local_tail_blocks[scale_id]
        if not isinstance(local_reader, nn.Identity):
            if tail_score_bias is not None:
                raise ValueError(
                    "R-conditioned local tail attention is incompatible with "
                    "global tail-score bias"
                )
            query_valid_map = (anchor_support[:, 0] > 0)
            return local_reader(
                anchor_memory,
                query_valid_map,
                tail_views,
                tail_valid_map,
            )
        tail_tokens, tail_valid = self._to_view_tokens(tail_views, tail_valid_map)
        anchor_query = anchor_memory.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        anchor_query_valid = (anchor_support > 0).permute(
            0, 2, 3, 1
        ).reshape(b * h * w)
        readout = self.tdom_memory_token_readouts[scale_id]
        two_token = isinstance(
            readout, (TwoTokenTailReadout, ThreeWayGatedTailReadout)
        )
        if two_token:
            # Do not sum or concatenate the modalities before attention.  At
            # every ground pixel, anchor and TDOM remain two independent
            # queries over exactly the same projected-image tail set.
            tdom_query = reference.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
            query = torch.cat((anchor_query, tdom_query), dim=1)
            # The TDOM token remains usable when a projected anchor is absent.
            # This is the central difference from treating TDOM as an additive
            # feature at the output: its relation to tails is independently
            # observable even at anchor gaps.
            tdom_query_valid = self._valid_pixels(
                reference.unsqueeze(1), view_exist=None
            ).permute(0, 2, 3, 1).reshape(b * h * w)
            query_valid = torch.stack((anchor_query_valid, tdom_query_valid), dim=1)
        else:
            query = anchor_query
            query_valid = anchor_query_valid.unsqueeze(1)
        key_padding_mask, empty_tail = _safe_key_padding_mask(tail_valid)
        tail_nonzero_map = self._valid_pixels(tail_views, tail_exist)
        _, tail_nonzero = self._to_view_tokens(tail_views, tail_nonzero_map)
        if self.tail_value_mode == "raw_weighted":
            if raw_tail_views is None:
                raise ValueError("raw_weighted requires pre-context raw_tail_views")
            raw_tokens, _ = self._to_view_tokens(raw_tail_views, tail_valid_map)
            attended = self._raw_weighted_tail_attention(
                query, tail_tokens, raw_tokens, tail_valid, query_valid, scale_id
            )
        elif not self.mask_zero_tail_features:
            attended, tail_acceptance = self._zero_tail_aware_attention(
                query,
                tail_tokens,
                tail_valid,
                tail_nonzero,
                query_valid,
                scale_id,
            )
        elif self.use_null_tail:
            attended, tail_acceptance = self._null_aware_tail_attention(
                query, tail_tokens, tail_valid, query_valid, scale_id
            )
        elif tail_score_bias is not None:
            attended = self._global_score_biased_tail_attention(
                query,
                tail_tokens,
                tail_valid,
                query_valid,
                tail_score_bias,
                scale_id,
                batch=b,
                height=h,
                width=w,
            )
        else:
            attended, _ = self.tail_attn[scale_id](
                self.tail_norm_q[scale_id](query),
                self.tail_norm_kv[scale_id](tail_tokens),
                self.tail_norm_kv[scale_id](tail_tokens),
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        delta = (attended if self.tail_value_mode == "raw_weighted" else
                 self._compose_anchor_tail_delta(query, attended, scale_id))
        if self.use_null_tail or not self.mask_zero_tail_features:
            # The FFN contains biases, so the zero null value alone is not
            # sufficient to guarantee a no-update path.  The real-tail
            # attention mass makes the whole residual update vanish when null
            # is selected.
            delta = delta * tail_acceptance[:, None, None]
        if self.mask_zero_tail_features:
            usable = query_valid & ~empty_tail.unsqueeze(1)
            support_valid = tail_valid
        else:
            # All-zero candidates are legitimate null exits, but cannot by
            # themselves create an update or claim image-feature support.
            usable = query_valid & tail_nonzero.any(dim=1, keepdim=True)
            support_valid = tail_nonzero
        delta = delta.masked_fill(~usable[:, :, None], 0.0)
        if isinstance(readout, ThreeWayGatedTailReadout):
            delta, _ = readout(delta, query, anchor_query_valid)
        elif isinstance(readout, TwoTokenTailReadout):
            delta, _ = readout(delta)
        else:
            delta = delta[:, :1]
        update = delta.reshape(b, h, w, c).permute(0, 3, 1, 2)
        count = support_valid.sum(dim=1).to(dtype=reference.dtype)
        support = count.reshape(b, 1, h, w) / float(max(q, 1))
        # The support map retains the existing anchor-support semantics.  A
        # TDOM-query update may exist at an anchor gap, but it must not make
        # the diagnostic anchor-support map falsely claim anchor evidence.
        support = support * anchor_query_valid.reshape(b, 1, h, w).to(reference.dtype)
        return update, support

    def _raw_weighted_tail_attention(
        self, query, key_tokens, raw_tokens, valid, query_valid, scale_id,
    ):
        """Average head weights, then pool entire unmodified raw vectors.

        Q/K are exactly the historical normalized/projected Q/K. No value
        normalization, Wv, Wo or FFN is applied to the pooled feature.
        """
        attention = self.tail_attn[scale_id]
        c, heads = attention.embed_dim, attention.num_heads
        nq = self.tail_norm_q[scale_id](query)
        nk = self.tail_norm_kv[scale_id](key_tokens)
        bias = attention.in_proj_bias
        qp = F.linear(nq, attention.in_proj_weight[:c], None if bias is None else bias[:c])
        kp = F.linear(nk, attention.in_proj_weight[c:2*c], None if bias is None else bias[c:2*c])
        n, q, _ = kp.shape
        qp = qp.reshape(n, 1, heads, c // heads).permute(0, 2, 1, 3)
        kp = kp.reshape(n, q, heads, c // heads).permute(0, 2, 1, 3)
        logits = torch.matmul(qp.float(), kp.float().transpose(-1, -2)) / math.sqrt(c // heads)
        usable = valid[:, None, None, :] & query_valid[:, None, :, None]
        logits = logits.masked_fill(~usable, -float("inf"))
        safe = torch.where(usable.any(-1, keepdim=True), logits, torch.zeros_like(logits))
        weights = torch.softmax(safe, -1) * usable.to(safe.dtype)
        return torch.matmul(weights.mean(1), raw_tokens.float()).to(query.dtype)

    def _zero_tail_aware_attention(
        self,
        query: torch.Tensor,
        tail_tokens: torch.Tensor,
        tail_valid: torch.Tensor,
        tail_nonzero: torch.Tensor,
        query_valid: torch.Tensor,
        scale_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Treat zero tails as strict null candidates.

        Existing zero tokens participate in the attention softmax through
        their keys, but never contribute a value.  The nonzero attention mass
        gates the complete residual later, preventing projection/FFN biases
        from producing an update when all tails are zero.
        """
        attention = self.tail_attn[scale_id]
        n, q, c = tail_tokens.shape
        if query.shape[1] != 1:
            raise ValueError("Zero-tail-aware attention currently expects one R query")
        if query_valid.ndim == 2:
            flat_query_valid = query_valid[:, 0]
        else:
            flat_query_valid = query_valid
        heads = attention.num_heads
        head_dim = c // heads
        norm_query = self.tail_norm_q[scale_id](query)
        norm_tail = self.tail_norm_kv[scale_id](tail_tokens)
        q_proj, k_proj, v_proj = self._mha_qkv_projection(
            attention, norm_query, norm_tail
        )
        q_proj = q_proj.reshape(n, 1, heads, head_dim).permute(0, 2, 1, 3)[:, :, 0]
        k_proj = k_proj.reshape(n, q, heads, head_dim).permute(0, 2, 1, 3)
        v_proj = v_proj.reshape(n, q, heads, head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum(
            "nhd,nhqd->nhq", q_proj.float(), k_proj.float()
        ) / math.sqrt(head_dim)
        usable = tail_valid[:, None, :] & flat_query_valid[:, None, None]
        logits = logits.masked_fill(~usable, -float("inf"))
        has_candidate = usable.any(dim=-1, keepdim=True)
        safe_logits = torch.where(has_candidate, logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=-1)
        weights = weights * usable.to(weights.dtype)

        nonzero_usable = usable & tail_nonzero[:, None, :]
        nonzero_weights = weights * nonzero_usable.to(weights.dtype)
        nonzero_mass = nonzero_weights.sum(dim=-1)
        conditional_weights = (
            nonzero_weights / nonzero_mass.clamp_min(self.eps).unsqueeze(-1)
        )
        context = torch.einsum(
            "nhq,nhqd->nhd", conditional_weights, v_proj.float()
        )
        context = context.reshape(n, 1, c).to(dtype=query.dtype)
        attended = F.linear(
            context, attention.out_proj.weight, attention.out_proj.bias
        )
        acceptance = nonzero_mass.mean(dim=1).to(dtype=query.dtype)
        return attended, acceptance

    def _global_score_biased_tail_attention(
        self,
        query: torch.Tensor,
        tail_tokens: torch.Tensor,
        tail_valid: torch.Tensor,
        query_valid: torch.Tensor,
        tail_score_bias: torch.Tensor,
        scale_id: int,
        *,
        batch: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """R-query attention with a differentiable patch-level tail prior."""
        attention = self.tail_attn[scale_id]
        n, q, c = tail_tokens.shape
        if tail_score_bias.shape != (batch, q):
            raise ValueError(
                f"tail_score_bias must be {(batch, q)}, got "
                f"{tuple(tail_score_bias.shape)}"
            )
        if n != batch * height * width:
            raise ValueError("Flattened token count does not match B*H*W")
        heads = attention.num_heads
        head_dim = c // heads
        norm_query = self.tail_norm_q[scale_id](query)
        norm_tail = self.tail_norm_kv[scale_id](tail_tokens)
        q_proj, k_proj, v_proj = self._mha_qkv_projection(
            attention, norm_query, norm_tail
        )
        q_proj = q_proj.reshape(n, 1, heads, head_dim).permute(0, 2, 1, 3)[:, :, 0]
        k_proj = k_proj.reshape(n, q, heads, head_dim).permute(0, 2, 1, 3)
        v_proj = v_proj.reshape(n, q, heads, head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum(
            "nhd,nhqd->nhq", q_proj.float(), k_proj.float()
        ) / math.sqrt(head_dim)
        spatial_bias = tail_score_bias[:, None, None, :].expand(
            batch, height, width, q
        ).reshape(n, q)
        logits = logits + spatial_bias[:, None].float() / self.tail_score_temperature
        usable = tail_valid[:, None, :] & query_valid[:, None, None]
        logits = logits.masked_fill(~usable, -float("inf"))
        has_tail = usable.any(dim=-1, keepdim=True)
        safe_logits = torch.where(has_tail, logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=-1)
        weights = weights * usable.to(weights.dtype)
        context = torch.einsum("nhq,nhqd->nhd", weights, v_proj.float())
        context = context.reshape(n, 1, c).to(dtype=query.dtype)
        return F.linear(
            context, attention.out_proj.weight, attention.out_proj.bias
        )

    def _null_aware_tail_attention(
        self,
        query: torch.Tensor,
        tail_tokens: torch.Tensor,
        tail_valid: torch.Tensor,
        query_valid: torch.Tensor,
        scale_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """R-query attention with an explicit zero-value null candidate.

        Returns the projected attention output and the mean attention mass
        assigned to real tails.  Scaling the residual by the latter makes the
        null candidate a genuine no-update route even in the presence of the
        tail FFN's bias terms.
        """
        attention = self.tail_attn[scale_id]
        n, q, c = tail_tokens.shape
        if query.shape[1] != 1:
            raise ValueError("Null-tail attention expects one anchor query")
        # The caller supplies [N, 1]; attention logits have shape [N, H, Q].
        query_valid = query_valid.reshape(n)
        heads = attention.num_heads
        head_dim = c // heads
        norm_query = self.tail_norm_q[scale_id](query)
        norm_tail = self.tail_norm_kv[scale_id](tail_tokens)
        q_proj, k_proj, v_proj = self._mha_qkv_projection(
            attention, norm_query, norm_tail
        )
        null_input = self.null_tail_keys[scale_id].to(
            device=query.device, dtype=query.dtype
        ).reshape(1, 1, c).expand(n, 1, c)
        norm_null = self.tail_norm_kv[scale_id](null_input)
        _, null_k, _ = self._mha_qkv_projection(attention, norm_query, norm_null)

        q_proj = q_proj.reshape(n, 1, heads, head_dim).permute(0, 2, 1, 3)[:, :, 0]
        k_proj = k_proj.reshape(n, q, heads, head_dim).permute(0, 2, 1, 3)
        v_proj = v_proj.reshape(n, q, heads, head_dim).permute(0, 2, 1, 3)
        null_k = null_k.reshape(n, 1, heads, head_dim).permute(0, 2, 1, 3)

        real_scores = torch.einsum(
            "nhd,nhqd->nhq", q_proj.float(), k_proj.float()
        ) / math.sqrt(head_dim)
        real_usable = tail_valid[:, None, :] & query_valid[:, None, None]
        real_scores = real_scores.masked_fill(~real_usable, -float("inf"))
        null_scores = torch.einsum(
            "nhd,nhqd->nhq", q_proj.float(), null_k.float()
        ) / math.sqrt(head_dim)
        null_scores = null_scores + self.null_tail_logit[scale_id].float()
        null_scores = null_scores.masked_fill(
            ~query_valid[:, None, None], -float("inf")
        )
        scores = torch.cat((real_scores, null_scores), dim=-1)

        # A pixel without anchor support has no valid query.  Use a harmless
        # temporary score there, then mask it out after softmax to avoid NaNs.
        scores = torch.where(
            query_valid[:, None, None], scores, torch.zeros_like(scores)
        )
        weights = torch.softmax(scores, dim=-1)
        weights = weights * query_valid[:, None, None].to(weights.dtype)
        real_weights = weights[..., :q]
        context = torch.einsum("nhq,nhqd->nhd", real_weights, v_proj.float())
        context = context.reshape(n, 1, c).to(dtype=query.dtype)
        attended = F.linear(context, attention.out_proj.weight, attention.out_proj.bias)
        tail_acceptance = real_weights.sum(dim=-1).mean(dim=1)
        return attended, tail_acceptance

    def _tailwise_candidates(
        self,
        query: torch.Tensor,
        tail_tokens: torch.Tensor,
        tail_valid: torch.Tensor,
        query_valid: torch.Tensor,
        scale_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce one residual proposal and score for every individual tail.

        Unlike view-axis attention, this step does not let one tail alter the
        representation of another.  The following Top-K pool is therefore the
        only place where tail evidence is combined.
        """
        if self.tailwise_update is None or self.tailwise_score is None:
            raise RuntimeError("tailwise candidates requested without tailwise_topk")
        n, q, _ = tail_tokens.shape
        query_tokens = self.tail_norm_q[scale_id](query).expand(-1, q, -1)
        tail_tokens = self.tail_norm_kv[scale_id](tail_tokens)
        pair = torch.cat(
            (query_tokens, tail_tokens, tail_tokens - query_tokens), dim=-1
        )
        candidates = self.tailwise_update[scale_id](pair)
        scores = self.tailwise_score[scale_id](pair).squeeze(-1)
        usable = tail_valid & query_valid[:, None]
        scores = scores.masked_fill(~usable, -float("inf"))
        return candidates, scores

    @staticmethod
    def _topk_tail_pool(
        candidates: torch.Tensor,
        scores: torch.Tensor,
        topk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparse, safe Top-K pool for ``[pixel, view, channel]`` candidates."""
        n, q, c = candidates.shape
        k = min(int(topk), q)
        top_scores, top_indices = torch.topk(scores, k=k, dim=1)
        top_candidates = torch.gather(
            candidates, 1, top_indices[..., None].expand(n, k, c)
        )
        has_real_tail = torch.isfinite(top_scores).any(dim=1)
        safe_scores = torch.where(
            has_real_tail[:, None], top_scores, torch.zeros_like(top_scores)
        )
        weights = torch.softmax(safe_scores, dim=1)
        weights = weights * torch.isfinite(top_scores).to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        update = (weights[..., None] * top_candidates).sum(dim=1, keepdim=True)
        update = update.masked_fill(~has_real_tail[:, None, None], 0.0)
        return update, has_real_tail, top_scores

    def _tailwise_topk_update(
        self,
        tail_views: Optional[torch.Tensor],
        tail_exist: Optional[torch.Tensor],
        anchor_memory: torch.Tensor,
        anchor_support: torch.Tensor,
        scale_id: int,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse aggregate of independently proposed tail updates."""
        b, c, h, w = reference.shape
        empty = torch.zeros_like(reference)
        empty_support = torch.zeros(
            (b, 1, h, w), device=reference.device, dtype=reference.dtype
        )
        if tail_views is None:
            return empty, empty_support
        bt, q, ct, ht, wt = tail_views.shape
        if (bt, ct, ht, wt) != (b, c, h, w):
            raise ValueError("Tail feature shape must match the TDOM feature at each scale")
        tail_valid_map = self._tail_attention_valid_pixels(tail_views, tail_exist)
        tail_tokens, tail_valid = self._to_view_tokens(tail_views, tail_valid_map)
        query = anchor_memory.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        query_valid = (anchor_support > 0).permute(0, 2, 3, 1).reshape(b * h * w)
        candidates, scores = self._tailwise_candidates(
            query, tail_tokens, tail_valid, query_valid, scale_id
        )
        update, has_real_tail, _ = self._topk_tail_pool(
            candidates, scores, self.tail_topk
        )
        update = update.masked_fill(~query_valid[:, None, None], 0.0)
        update = update.reshape(b, h, w, c).permute(0, 3, 1, 2)
        count = tail_valid.sum(dim=1).to(dtype=reference.dtype)
        support = count.reshape(b, 1, h, w) / float(max(q, 1))
        support = support * query_valid.reshape(b, 1, h, w).to(reference.dtype)
        support = support * has_real_tail.reshape(b, 1, h, w).to(reference.dtype)
        return update, support

    @staticmethod
    def _mha_qkv_projection(
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Q/K/V with the parameters of a standard same-dim MHA."""
        if attention.in_proj_weight is None:
            raise ValueError("Streaming R-query attention requires packed MHA weights")
        embed_dim = attention.embed_dim
        wq, wk, wv = attention.in_proj_weight.split(embed_dim, dim=0)
        if attention.in_proj_bias is None:
            bq = bk = bv = None
        else:
            bq, bk, bv = attention.in_proj_bias.split(embed_dim, dim=0)
        return (
            F.linear(query, wq, bq),
            F.linear(key_value, wk, bk),
            F.linear(key_value, wv, bv),
        )

    def _forward_streaming_anchor_query(
        self,
        tdom_features: Sequence[torch.Tensor],
        *,
        anchor_features,
        anchor_exist: Optional[torch.Tensor],
        tail_feature_chunks,
        output_attn: bool,
        return_aux: bool,
    ):
        """Exact all-tail R-query attention using online softmax reduction."""
        states = []
        for scale_id, (tdom, scale) in enumerate(zip(tdom_features, _SCALES)):
            anchors = self._stack_views(anchor_features, scale, tdom.device, tdom.dtype)
            anchor_memory, anchor_support, _, _, swin_alpha, delta_ratio, latent_cos, latent_entropy, weight_mean = self._anchor_memory(
                anchors, anchor_exist, scale_id, tdom
            )
            b, c, h, w = tdom.shape
            n = b * h * w
            local_reader = self.r_conditioned_local_tail_blocks[scale_id]
            if not isinstance(local_reader, nn.Identity):
                query_valid_map = anchor_support[:, 0] > 0
                states.append({
                    "local_reader": True,
                    "local_state": local_reader.init_stream(
                        anchor_memory, query_valid_map
                    ),
                    "anchor_memory": anchor_memory,
                    "anchor_support": anchor_support,
                    "swin_r_alpha": swin_alpha,
                    "swin_r_delta_ratio": delta_ratio,
                    "latent_pairwise_cos": latent_cos,
                    "latent_weight_entropy": latent_entropy,
                    "latent_view_weight_mean": weight_mean,
                    "tail_context_alpha_sum": torch.zeros(
                        (), device=tdom.device, dtype=torch.float32
                    ),
                    "tail_context_ratio_sum": torch.zeros(
                        (), device=tdom.device, dtype=torch.float32
                    ),
                    "tail_context_chunk_count": 0,
                })
                continue
            heads = self.tail_attn[scale_id].num_heads
            head_dim = c // heads
            query = anchor_memory.permute(0, 2, 3, 1).reshape(n, 1, c)
            query_valid = (anchor_support > 0).permute(0, 2, 3, 1).reshape(n)
            maximum = torch.full(
                (n, heads), -float("inf"), device=tdom.device,
                dtype=torch.float32,
            )
            denominator = torch.zeros(
                (n, heads), device=tdom.device, dtype=torch.float32
            )
            if self.use_null_tail:
                attention = self.tail_attn[scale_id]
                norm_query = self.tail_norm_q[scale_id](query)
                null_input = self.null_tail_keys[scale_id].to(
                    device=tdom.device, dtype=tdom.dtype
                ).reshape(1, 1, c).expand(n, 1, c)
                norm_null = self.tail_norm_kv[scale_id](null_input)
                q_proj, null_k, _ = self._mha_qkv_projection(
                    attention, norm_query, norm_null
                )
                q_proj = q_proj.reshape(n, 1, heads, head_dim).permute(
                    0, 2, 1, 3
                )[:, :, 0]
                null_k = null_k.reshape(n, 1, heads, head_dim).permute(0, 2, 1, 3)
                null_score = torch.einsum(
                    "nhd,nhqd->nhq", q_proj.float(), null_k.float()
                )[:, :, 0] / math.sqrt(head_dim)
                null_score = null_score + self.null_tail_logit[scale_id].float()
                maximum = null_score.masked_fill(~query_valid[:, None], -float("inf"))
                denominator = query_valid[:, None].to(torch.float32).expand(-1, heads).clone()
            states.append({
                "local_reader": False,
                "anchor_memory": anchor_memory,
                "anchor_support": anchor_support,
                "swin_r_alpha": swin_alpha,
                "swin_r_delta_ratio": delta_ratio,
                "latent_pairwise_cos": latent_cos,
                "latent_weight_entropy": latent_entropy,
                "latent_view_weight_mean": weight_mean,
                "query": query,
                "query_valid": query_valid,
                "maximum": maximum,
                "denominator": denominator,
                "weighted_value": torch.zeros(
                    (n, heads, c if self.tail_value_mode == "raw_weighted" else head_dim),
                    device=tdom.device, dtype=torch.float32
                ),
                "tail_count": torch.zeros(
                    (n,), device=tdom.device, dtype=torch.float32
                ),
                "tail_weight": torch.zeros(
                    (n, heads), device=tdom.device, dtype=torch.float32
                ),
                # A stream may contain arbitrary-size tail chunks.  Keep
                # scalar diagnostics online as well, so all-tail prediction
                # reports the same TailLocal statistics as fixed-Q forward.
                "tail_context_alpha_sum": torch.zeros(
                    (), device=tdom.device, dtype=torch.float32
                ),
                "tail_context_ratio_sum": torch.zeros(
                    (), device=tdom.device, dtype=torch.float32
                ),
                "tail_context_chunk_count": 0,
            })

        total_tail_slots = 0
        for chunk_features, chunk_exist in tail_feature_chunks:
            if not chunk_features:
                continue
            chunk_slots = len(chunk_features)
            total_tail_slots += chunk_slots
            for scale_id, (tdom, scale) in enumerate(zip(tdom_features, _SCALES)):
                state = states[scale_id]
                tails = self._stack_views(chunk_features, scale, tdom.device, tdom.dtype)
                raw_tails = tails
                # The all-tail online softmax is exact only if a tail has the
                # same representation regardless of which stream chunk it is
                # placed in.  TailLocal is independent per tail, so applying
                # it here before tokenisation preserves that property while
                # keeping the stream memory-bounded.
                tails, tail_context_alpha, tail_context_ratio = (
                    self._contextualize_tail_views(
                        tails, chunk_exist, scale_id
                    )
                )
                state["tail_context_alpha_sum"] += tail_context_alpha.float()
                state["tail_context_ratio_sum"] += tail_context_ratio.float()
                state["tail_context_chunk_count"] += 1
                tail_valid_map = self._tail_attention_valid_pixels(tails, chunk_exist)
                if state["local_reader"]:
                    local_reader = self.r_conditioned_local_tail_blocks[scale_id]
                    state["local_state"] = local_reader.accumulate_stream(
                        state["local_state"], tails, tail_valid_map
                    )
                    continue
                tail_tokens, tail_valid = self._to_view_tokens(tails, tail_valid_map)
                tail_nonzero_map = self._valid_pixels(tails, chunk_exist)
                _, tail_nonzero = self._to_view_tokens(tails, tail_nonzero_map)
                norm_query = self.tail_norm_q[scale_id](state["query"])
                norm_tail = self.tail_norm_kv[scale_id](tail_tokens)
                q_proj, k_proj, v_proj = self._mha_qkv_projection(
                    self.tail_attn[scale_id], norm_query, norm_tail
                )
                n, q, c = k_proj.shape
                heads = self.tail_attn[scale_id].num_heads
                head_dim = c // heads
                q_proj = q_proj.reshape(n, 1, heads, head_dim).permute(0, 2, 1, 3)[:, :, 0]
                k_proj = k_proj.reshape(n, q, heads, head_dim).permute(0, 2, 1, 3)
                v_proj = v_proj.reshape(n, q, heads, head_dim).permute(0, 2, 1, 3)
                if self.tail_value_mode == "raw_weighted":
                    raw_tokens, _ = self._to_view_tokens(raw_tails, tail_valid_map)
                    v_proj = raw_tokens[:, None].expand(-1, heads, -1, -1)
                scores = torch.einsum(
                    "nhd,nhqd->nhq", q_proj.float(), k_proj.float()
                ) / math.sqrt(head_dim)
                usable = (
                    tail_valid[:, None, :]
                    & state["query_valid"][:, None, None]
                )
                scores = scores.masked_fill(~usable, -float("inf"))
                chunk_max = scores.max(dim=-1).values
                old_has = state["denominator"] > 0
                chunk_has = usable.any(dim=-1).expand(-1, heads)
                combined_max = torch.maximum(state["maximum"], chunk_max)
                new_max = torch.where(
                    old_has | chunk_has, combined_max, torch.zeros_like(combined_max)
                )
                old_factor = torch.where(
                    old_has,
                    torch.exp(state["maximum"] - new_max),
                    torch.zeros_like(new_max),
                )
                weights = torch.where(
                    usable,
                    torch.exp(scores - new_max.unsqueeze(-1)),
                    torch.zeros_like(scores),
                )
                if self.mask_zero_tail_features:
                    value_weights = weights
                    support_valid = tail_valid
                else:
                    # Zero tails stay in the softmax denominator as null
                    # exits, but cannot contribute a projected Value.
                    value_weights = weights * tail_nonzero[:, None, :].to(
                        weights.dtype
                    )
                    support_valid = tail_nonzero
                state["weighted_value"] = (
                    state["weighted_value"] * old_factor.unsqueeze(-1)
                    + torch.einsum(
                        "nhq,nhqd->nhd", value_weights, v_proj.float()
                    )
                )
                state["denominator"] = (
                    state["denominator"] * old_factor + weights.sum(dim=-1)
                )
                state["tail_weight"] = (
                    state["tail_weight"] * old_factor
                    + value_weights.sum(dim=-1)
                )
                state["maximum"] = new_max
                state["tail_count"] += support_valid.sum(dim=1).float()

        fused = []
        aux: Dict[str, list] = {
            "anchor_memory": [], "tail_memory": [],
            "anchor_support": [], "tail_support": [],
            "alpha": [], "tail_beta": [],
            "swin_r_alpha": [], "swin_r_delta_ratio": [],
            "latent_pairwise_cos": [], "latent_weight_entropy": [],
            "latent_view_weight_mean": [],
            "tail_local_context_alpha": [],
            "tail_local_context_delta_ratio": [],
            "tail_dual_memory_context_alpha": [],
            "tail_dual_memory_context_delta_ratio": [],
        }
        for scale_id, tdom in enumerate(tdom_features):
            state = states[scale_id]
            if state["local_reader"]:
                local_reader = self.r_conditioned_local_tail_blocks[scale_id]
                tail_update, tail_support = local_reader.finalize_stream(
                    state["local_state"], total_tail_slots
                )
            else:
                attention = self.tail_attn[scale_id]
                denominator = state["denominator"]
                if self.mask_zero_tail_features:
                    value_denominator = denominator
                else:
                    value_denominator = state["tail_weight"]
                context = state["weighted_value"] / value_denominator.clamp_min(
                    self.eps
                ).unsqueeze(-1)
                if self.tail_value_mode == "raw_weighted":
                    context = context.mean(dim=1, keepdim=True)
                else:
                    context = context.reshape(context.shape[0], 1, attention.embed_dim)
                context = context.to(dtype=state["query"].dtype)
                if self.tail_value_mode == "raw_weighted":
                    delta = context
                else:
                    attended = F.linear(
                        context, attention.out_proj.weight, attention.out_proj.bias
                    )
                    delta = self._compose_anchor_tail_delta(state["query"], attended, scale_id)
                if self.use_null_tail or not self.mask_zero_tail_features:
                    tail_acceptance = (
                        state["tail_weight"] / denominator.clamp_min(self.eps)
                    ).mean(dim=1)
                    delta = delta * tail_acceptance[:, None, None].to(delta.dtype)
                if self.mask_zero_tail_features:
                    has_evidence = denominator.sum(dim=1) > 0
                else:
                    has_evidence = state["tail_weight"].sum(dim=1) > 0
                usable = state["query_valid"] & has_evidence
                delta = delta.masked_fill(~usable[:, None, None], 0.0)
                b, c, h, w = tdom.shape
                tail_update = delta.reshape(b, h, w, c).permute(0, 3, 1, 2)
                tail_support = state["tail_count"].reshape(b, 1, h, w)
                tail_support = tail_support / float(max(total_tail_slots, 1))
                tail_support = tail_support * state["query_valid"].reshape(
                    b, 1, h, w
                ).to(tdom.dtype)
            beta = self._effective_tail_beta(scale_id, tdom)
            alpha = self.alpha[scale_id].to(dtype=tdom.dtype, device=tdom.device)
            memory = self._merge_anchor_tail(state["anchor_memory"], tail_update, beta)
            fused.append(self._fuse_tdom_memory(tdom, memory, scale_id))
            aux["anchor_memory"].append(state["anchor_memory"])
            aux["tail_memory"].append(tail_update)
            aux["anchor_support"].append(state["anchor_support"])
            aux["tail_support"].append(tail_support)
            aux["alpha"].append(alpha.detach())
            aux["tail_beta"].append(beta.detach())
            aux["swin_r_alpha"].append(state["swin_r_alpha"])
            aux["swin_r_delta_ratio"].append(state["swin_r_delta_ratio"])
            aux["latent_pairwise_cos"].append(state["latent_pairwise_cos"])
            aux["latent_weight_entropy"].append(state["latent_weight_entropy"])
            aux["latent_view_weight_mean"].append(state["latent_view_weight_mean"])
            context_count = max(state["tail_context_chunk_count"], 1)
            aux["tail_local_context_alpha"].append(
                state["tail_context_alpha_sum"] / float(context_count)
            )
            aux["tail_local_context_delta_ratio"].append(
                state["tail_context_ratio_sum"] / float(context_count)
            )

        if output_attn and return_aux:
            return fused, None, aux
        if output_attn:
            return fused, None
        if return_aux:
            return fused, aux
        return fused

    def forward_multi_fixed_q(
        self,
        tdom_features: Sequence[torch.Tensor],
        *,
        anchor_features,
        anchor_exist: Optional[torch.Tensor],
        tail_features,
        tail_exist: Optional[torch.Tensor],
        q_values: Sequence[int],
    ) -> Dict[int, Sequence[torch.Tensor]]:
        """Fuse several prefix-Q subsets from one already-loaded tail batch.

        This is an inference-only convenience path for Q-sweep experiments.
        The caller supplies one deterministic, nested tail list padded to the
        largest Q.  We reuse the encoder/neck output and the loaded feature
        tensors, while recomputing only the SAIL fusion for each prefix.  It
        deliberately does not rank or resample tails, so it preserves the
        fixed nested-view experiment semantics.
        """
        if not tail_features:
            raise ValueError("forward_multi_fixed_q requires loaded tail features")
        if self.tail_value_mode == "raw_weighted":
            raise NotImplementedError("raw_weighted currently uses fixed-Q or all-tail streaming, not multi-Q one-pass")
        values = sorted({int(value) for value in q_values})
        if not values or values[0] < 1:
            raise ValueError("q_values must contain positive integers")
        if values[-1] > len(tail_features):
            raise ValueError(
                f"q_values exceed loaded tail slots: max={values[-1]}, "
                f"loaded={len(tail_features)}"
            )
        outputs: Dict[int, Sequence[torch.Tensor]] = {}
        for value in values:
            selected_features = tail_features[:value]
            selected_exist = (
                tail_exist[:, :value] if tail_exist is not None else None
            )
            outputs[value] = self.forward(
                tdom_features,
                anchor_features=anchor_features,
                anchor_exist=anchor_exist,
                tail_features=selected_features,
                tail_exist=selected_exist,
            )
        return outputs

    def _forward_streaming_tailwise_topk(
        self,
        tdom_features: Sequence[torch.Tensor],
        *,
        anchor_features,
        anchor_exist: Optional[torch.Tensor],
        tail_feature_chunks,
        output_attn: bool,
        return_aux: bool,
    ):
        """Exact streaming Top-K pool over every available tail view.

        Only the current best ``K`` candidates per ground pixel are retained,
        so all-tail inference remains bounded even when a patch has many
        projected image features.
        """
        states = []
        for scale_id, (tdom, scale) in enumerate(zip(tdom_features, _SCALES)):
            anchors = self._stack_views(anchor_features, scale, tdom.device, tdom.dtype)
            anchor_memory, anchor_support, _, _, swin_alpha, delta_ratio, latent_cos, latent_entropy, weight_mean = self._anchor_memory(
                anchors, anchor_exist, scale_id, tdom
            )
            b, c, h, w = tdom.shape
            n = b * h * w
            states.append({
                "anchor_memory": anchor_memory,
                "anchor_support": anchor_support,
                "swin_r_alpha": swin_alpha,
                "swin_r_delta_ratio": delta_ratio,
                "latent_pairwise_cos": latent_cos,
                "latent_weight_entropy": latent_entropy,
                "latent_view_weight_mean": weight_mean,
                "query": anchor_memory.permute(0, 2, 3, 1).reshape(n, 1, c),
                "query_valid": (anchor_support > 0).permute(0, 2, 3, 1).reshape(n),
                "top_scores": torch.full(
                    (n, self.tail_topk), -float("inf"), device=tdom.device,
                    dtype=torch.float32,
                ),
                "top_updates": torch.zeros(
                    (n, self.tail_topk, c), device=tdom.device, dtype=tdom.dtype,
                ),
                "tail_count": torch.zeros(
                    (n,), device=tdom.device, dtype=torch.float32
                ),
            })

        total_tail_slots = 0
        for chunk_features, chunk_exist in tail_feature_chunks:
            if not chunk_features:
                continue
            total_tail_slots += len(chunk_features)
            for scale_id, (tdom, scale) in enumerate(zip(tdom_features, _SCALES)):
                state = states[scale_id]
                tails = self._stack_views(chunk_features, scale, tdom.device, tdom.dtype)
                tail_valid_map = self._tail_attention_valid_pixels(tails, chunk_exist)
                tail_tokens, tail_valid = self._to_view_tokens(tails, tail_valid_map)
                candidates, scores = self._tailwise_candidates(
                    state["query"], tail_tokens, tail_valid,
                    state["query_valid"], scale_id,
                )
                candidate_scores = torch.cat((state["top_scores"], scores.float()), dim=1)
                candidate_updates = torch.cat((state["top_updates"], candidates), dim=1)
                top_scores, top_indices = torch.topk(
                    candidate_scores, k=self.tail_topk, dim=1
                )
                state["top_scores"] = top_scores
                state["top_updates"] = torch.gather(
                    candidate_updates, 1,
                    top_indices[..., None].expand(-1, -1, candidate_updates.shape[-1]),
                )
                state["tail_count"] += tail_valid.sum(dim=1).float()

        fused = []
        aux: Dict[str, list] = {
            "anchor_memory": [], "tail_memory": [],
            "anchor_support": [], "tail_support": [],
            "alpha": [], "tail_beta": [],
            "swin_r_alpha": [], "swin_r_delta_ratio": [],
            "latent_pairwise_cos": [], "latent_weight_entropy": [],
            "latent_view_weight_mean": [],
        }
        for scale_id, tdom in enumerate(tdom_features):
            state = states[scale_id]
            update, has_real_tail, _ = self._topk_tail_pool(
                state["top_updates"], state["top_scores"], self.tail_topk
            )
            b, c, h, w = tdom.shape
            tail_update = update.reshape(b, h, w, c).permute(0, 3, 1, 2)
            tail_support = state["tail_count"].reshape(b, 1, h, w)
            tail_support = tail_support / float(max(total_tail_slots, 1))
            tail_support = tail_support * state["query_valid"].reshape(
                b, 1, h, w
            ).to(tdom.dtype)
            tail_support = tail_support * has_real_tail.reshape(
                b, 1, h, w
            ).to(tdom.dtype)
            beta = self._effective_tail_beta(scale_id, tdom)
            alpha = self.alpha[scale_id].to(dtype=tdom.dtype, device=tdom.device)
            memory = self._merge_anchor_tail(state["anchor_memory"], tail_update, beta)
            fused.append(self._fuse_tdom_memory(tdom, memory, scale_id))
            aux["anchor_memory"].append(state["anchor_memory"])
            aux["tail_memory"].append(tail_update)
            aux["anchor_support"].append(state["anchor_support"])
            aux["tail_support"].append(tail_support)
            aux["alpha"].append(alpha.detach())
            aux["tail_beta"].append(beta.detach())
            aux["swin_r_alpha"].append(state["swin_r_alpha"])
            aux["swin_r_delta_ratio"].append(state["swin_r_delta_ratio"])
            aux["latent_pairwise_cos"].append(state["latent_pairwise_cos"])
            aux["latent_weight_entropy"].append(state["latent_weight_entropy"])
            aux["latent_view_weight_mean"].append(state["latent_view_weight_mean"])

        if output_attn and return_aux:
            return fused, None, aux
        if output_attn:
            return fused, None
        if return_aux:
            return fused, aux
        return fused

    def _select_streaming_global_tail_features(
        self,
        tdom_features: Sequence[torch.Tensor],
        *,
        anchor_features,
        anchor_exist: Optional[torch.Tensor],
        tail_feature_chunks,
        keep_limit: int,
    ):
        """Score every on-disk tail once and retain the global best views.

        Only the selected feature dictionaries are retained on CPU.  Thus the
        ranking is exact across all chunks while GPU memory stays bounded by
        the chunk size during scoring.
        """
        ref_id = self.tail_global_score_scale_id
        ref_scale = self.tail_global_score_scale
        ref_tdom = tdom_features[ref_id]
        anchors = self._stack_views(
            anchor_features, ref_scale, ref_tdom.device, ref_tdom.dtype
        )
        ref_memory, ref_support = self._anchor_memory(
            anchors, anchor_exist, ref_id, ref_tdom
        )[:2]
        kept_features = []
        kept_scores = []
        keep_limit = min(int(keep_limit), self.tail_topk)
        for chunk_features, chunk_exist in tail_feature_chunks:
            if not chunk_features:
                continue
            tails = self._stack_views(
                chunk_features, ref_scale, ref_tdom.device, ref_tdom.dtype
            )
            scores = self._global_tail_scores(
                tails,
                chunk_exist.to(device=ref_tdom.device),
                ref_memory,
                ref_support,
            )
            # All-tail prediction is intentionally batch_size=1.
            for score, feature in zip(scores[0].detach().cpu().tolist(), chunk_features):
                if math.isfinite(score) and score > -9999.0:
                    kept_scores.append(float(score))
                    kept_features.append(feature)
            if len(kept_features) > keep_limit:
                order = sorted(
                    range(len(kept_scores)),
                    key=kept_scores.__getitem__,
                    reverse=True,
                )[:keep_limit]
                kept_scores = [kept_scores[index] for index in order]
                kept_features = [kept_features[index] for index in order]

        if kept_features:
            order = sorted(
                range(len(kept_scores)),
                key=kept_scores.__getitem__,
                reverse=True,
            )
            kept_features = [kept_features[index] for index in order]
        return kept_features

    def _forward_streaming_global_tail_topk(
        self,
        tdom_features: Sequence[torch.Tensor],
        *,
        anchor_features,
        anchor_exist: Optional[torch.Tensor],
        tail_feature_chunks,
        output_attn: bool,
        return_aux: bool,
    ):
        """Score every on-disk tail, then fuse the global best K views."""
        kept_features = self._select_streaming_global_tail_features(
            tdom_features,
            anchor_features=anchor_features,
            anchor_exist=anchor_exist,
            tail_feature_chunks=tail_feature_chunks,
            keep_limit=self.tail_topk_eval,
        )
        if kept_features:
            ref_tdom = tdom_features[self.tail_global_score_scale_id]
            tail_exist = torch.ones(
                (1, len(kept_features)), dtype=torch.bool,
                device=ref_tdom.device,
            )
        else:
            tail_exist = None
        return self.forward(
            tdom_features,
            anchor_features=anchor_features,
            anchor_exist=anchor_exist,
            tail_features=kept_features,
            tail_exist=tail_exist,
            output_attn=output_attn,
            return_aux=return_aux,
        )

    def forward_streaming_multi_topk(
        self,
        tdom_features: Sequence[torch.Tensor],
        *,
        anchor_features,
        anchor_exist: Optional[torch.Tensor],
        tail_feature_chunks,
        topk_values: Sequence[int],
    ) -> Dict[int, Sequence[torch.Tensor]]:
        """Produce several Top-K fusions after one all-tail load/score pass."""
        if self.tail_aggregation != "global_tail_topk":
            raise ValueError("multi-TopK streaming requires global_tail_topk")
        values = sorted({int(value) for value in topk_values})
        if not values or values[0] < 1 or values[-1] > self.tail_topk:
            raise ValueError(
                f"topk_values must be within [1, {self.tail_topk}]"
            )
        kept_features = self._select_streaming_global_tail_features(
            tdom_features,
            anchor_features=anchor_features,
            anchor_exist=anchor_exist,
            tail_feature_chunks=tail_feature_chunks,
            keep_limit=values[-1],
        )
        ref_tdom = tdom_features[self.tail_global_score_scale_id]
        original_eval_k = self.tail_topk_eval
        self.tail_topk_eval = values[-1]
        outputs = {}
        try:
            for value in values:
                selected = kept_features[:value]
                tail_exist = (
                    torch.ones(
                        (1, len(selected)), dtype=torch.bool,
                        device=ref_tdom.device,
                    )
                    if selected else None
                )
                outputs[value] = self.forward(
                    tdom_features,
                    anchor_features=anchor_features,
                    anchor_exist=anchor_exist,
                    tail_features=selected,
                    tail_exist=tail_exist,
                )
        finally:
            self.tail_topk_eval = original_eval_k
        return outputs

    def forward_streaming(
        self,
        tdom_features: Sequence[torch.Tensor],
        *,
        anchor_features,
        anchor_exist: Optional[torch.Tensor],
        tail_feature_chunks,
        output_attn: bool = False,
        return_aux: bool = False,
    ):
        """Fuse every tail view from a bounded-memory chunk iterator.

        ``tail_feature_chunks`` yields ``(tail_features, tail_exist)`` where
        each feature collection follows the normal list-of-scale-dicts format.
        There is deliberately no tail-to-tail attention, so each chunk can be
        processed independently and its calibrated sum/count accumulated.
        """
        if self.tdom_memory_token_enabled:
            raise NotImplementedError(
                "Two-token R currently supports fixed-Q training/inference. "
                "Its all-tail online-softmax reducer must retain separate "
                "normalizers for the anchor and TDOM queries."
            )
        if len(tdom_features) > self.num_scales:
            raise ValueError(
                f"Got {len(tdom_features)} TDOM scales but module has {self.num_scales}"
            )
        if (
            self.attention_direction == "anchor_to_tail"
            and self.tail_aggregation == "global_tail_topk"
        ):
            return self._forward_streaming_global_tail_topk(
                tdom_features,
                anchor_features=anchor_features,
                anchor_exist=anchor_exist,
                tail_feature_chunks=tail_feature_chunks,
                output_attn=output_attn,
                return_aux=return_aux,
            )
        if (
            self.attention_direction == "anchor_to_tail"
            and self.tail_aggregation == "tailwise_topk"
        ):
            return self._forward_streaming_tailwise_topk(
                tdom_features,
                anchor_features=anchor_features,
                anchor_exist=anchor_exist,
                tail_feature_chunks=tail_feature_chunks,
                output_attn=output_attn,
                return_aux=return_aux,
            )
        if self.attention_direction == "anchor_to_tail":
            return self._forward_streaming_anchor_query(
                tdom_features,
                anchor_features=anchor_features,
                anchor_exist=anchor_exist,
                tail_feature_chunks=tail_feature_chunks,
                output_attn=output_attn,
                return_aux=return_aux,
            )

        prepared = []
        for scale_id, (tdom, scale) in enumerate(zip(tdom_features, _SCALES)):
            anchors = self._stack_views(anchor_features, scale, tdom.device, tdom.dtype)
            anchor_memory, anchor_support, anchor_tokens, anchor_valid, swin_alpha, delta_ratio, latent_cos, latent_entropy, weight_mean = self._anchor_memory(
                anchors, anchor_exist, scale_id, tdom
            )
            prepared.append((anchor_memory, anchor_support, anchor_tokens, anchor_valid, swin_alpha, delta_ratio, latent_cos, latent_entropy, weight_mean))

        tail_sums = [torch.zeros_like(tdom) for tdom in tdom_features]
        tail_counts = [
            torch.zeros((tdom.shape[0], 1, *tdom.shape[-2:]), device=tdom.device, dtype=tdom.dtype)
            for tdom in tdom_features
        ]
        total_tail_slots = 0
        for chunk_features, chunk_exist in tail_feature_chunks:
            if not chunk_features:
                continue
            total_tail_slots += len(chunk_features)
            for scale_id, (tdom, scale) in enumerate(zip(tdom_features, _SCALES)):
                tails = self._stack_views(chunk_features, scale, tdom.device, tdom.dtype)
                _, _, anchor_tokens, anchor_valid, _, _, _, _, _ = prepared[scale_id]
                chunk_sum, chunk_count, _ = self._tail_memory_sum(
                    tails, chunk_exist, anchor_tokens, anchor_valid, scale_id, tdom
                )
                tail_sums[scale_id] += chunk_sum
                tail_counts[scale_id] += chunk_count

        fused = []
        aux: Dict[str, list] = {
            "anchor_memory": [], "tail_memory": [],
            "anchor_support": [], "tail_support": [],
            "alpha": [], "tail_beta": [],
            "swin_r_alpha": [], "swin_r_delta_ratio": [],
            "latent_pairwise_cos": [], "latent_weight_entropy": [],
            "latent_view_weight_mean": [],
        }
        for scale_id, tdom in enumerate(tdom_features):
            anchor_memory, anchor_support, _, _, swin_alpha, delta_ratio, latent_cos, latent_entropy, weight_mean = prepared[scale_id]
            tail_memory = tail_sums[scale_id] / tail_counts[scale_id].clamp_min(1.0)
            tail_memory = torch.where(tail_counts[scale_id] > 0, tail_memory, torch.zeros_like(tail_memory))
            tail_support = tail_counts[scale_id] / float(max(total_tail_slots, 1))
            beta = self._effective_tail_beta(scale_id, tdom)
            alpha = self.alpha[scale_id].to(dtype=tdom.dtype, device=tdom.device)
            memory = self._merge_anchor_tail(anchor_memory, tail_memory, beta, conditioned_path=True)
            fused.append(self._fuse_tdom_memory(tdom, memory, scale_id))
            aux["anchor_memory"].append(anchor_memory)
            aux["tail_memory"].append(tail_memory)
            aux["anchor_support"].append(anchor_support)
            aux["tail_support"].append(tail_support)
            aux["alpha"].append(alpha.detach())
            aux["tail_beta"].append(beta.detach())
            aux["swin_r_alpha"].append(swin_alpha)
            aux["swin_r_delta_ratio"].append(delta_ratio)
            aux["latent_pairwise_cos"].append(latent_cos)
            aux["latent_weight_entropy"].append(latent_entropy)
            aux["latent_view_weight_mean"].append(weight_mean)

        if output_attn and return_aux:
            return fused, None, aux
        if output_attn:
            return fused, None
        if return_aux:
            return fused, aux
        return fused

    def forward(
        self,
        tdom_features: Sequence[torch.Tensor],
        cam_features=None,
        cam_exist: Optional[torch.Tensor] = None,
        *,
        anchor_features=None,
        anchor_exist: Optional[torch.Tensor] = None,
        tail_features=None,
        tail_exist: Optional[torch.Tensor] = None,
        output_attn: bool = False,
        return_aux: bool = False,
    ):
        if len(tdom_features) > self.num_scales:
            raise ValueError(
                f"Got {len(tdom_features)} TDOM scales but module has {self.num_scales}"
            )
        # Compatibility path for the existing collate/dataset: its cameras are anchors.
        if anchor_features is None:
            anchor_features = cam_features
            anchor_exist = cam_exist if anchor_exist is None else anchor_exist

        prepared = []
        for scale_id, (tdom, scale) in enumerate(zip(tdom_features, _SCALES)):
            anchors = self._stack_views(
                anchor_features, scale, tdom.device, tdom.dtype
            )
            prepared.append(
                self._anchor_memory(
                    anchors, anchor_exist, scale_id, tdom
                )
            )

        selected_indices = None
        selected_scores = None
        selected_tail_exist = tail_exist
        if self.tail_aggregation == "global_tail_topk" and tail_features:
            ref_id = self.tail_global_score_scale_id
            ref_tdom = tdom_features[ref_id]
            ref_tails = self._stack_views(
                tail_features,
                self.tail_global_score_scale,
                ref_tdom.device,
                ref_tdom.dtype,
            )
            ref_tails, _, _ = self._contextualize_tail_views(
                ref_tails, tail_exist, ref_id
            )
            ref_memory, ref_support = prepared[ref_id][:2]
            global_scores = self._global_tail_scores(
                ref_tails, tail_exist, ref_memory, ref_support
            )
            k = self._active_tail_topk(ref_tails.shape[1])
            if k > 0:
                selected_scores, selected_indices = torch.topk(
                    global_scores, k=k, dim=1
                )
                if tail_exist is not None:
                    selected_tail_exist = self._gather_tail_axis(
                        tail_exist.to(selected_indices.device), selected_indices
                    )
            del ref_tails

        fused = []
        aux: Dict[str, list] = {
            "anchor_memory": [], "tail_memory": [],
            "anchor_support": [], "tail_support": [],
            "alpha": [], "tail_beta": [],
            "swin_r_alpha": [], "swin_r_delta_ratio": [],
            "latent_pairwise_cos": [], "latent_weight_entropy": [],
            "latent_view_weight_mean": [],
            "tail_local_context_alpha": [],
            "tail_local_context_delta_ratio": [],
            "tail_dual_memory_context_alpha": [],
            "tail_dual_memory_context_delta_ratio": [],
        }
        for scale_id, (tdom, scale) in enumerate(zip(tdom_features, _SCALES)):
            if selected_indices is not None:
                tails = self._stack_selected_views(
                    tail_features,
                    scale,
                    selected_indices,
                    tdom.device,
                    tdom.dtype,
                )
            else:
                tails = self._stack_views(
                    tail_features, scale, tdom.device, tdom.dtype
                )
            active_tail_exist = (
                selected_tail_exist if selected_indices is not None else tail_exist
            )
            anchor_memory, anchor_support, anchor_tokens, anchor_valid, swin_alpha, delta_ratio, latent_cos, latent_entropy, weight_mean = prepared[scale_id]
            raw_tails = tails
            if self.tail_aggregation == "direct_mean":
                zero = tdom.new_zeros(())
                tail_context_alpha = zero
                tail_context_ratio = zero
                tail_dual_memory_alpha = zero
                tail_dual_memory_ratio = zero
                tail_memory, tail_support = self._direct_tail_mean(
                    tails, active_tail_exist, tdom
                )
            else:
                tails, tail_context_alpha, tail_context_ratio = self._contextualize_tail_views(
                    tails, active_tail_exist, scale_id
                )
                tails, tail_dual_memory_alpha, tail_dual_memory_ratio = (
                    self._condition_tails_on_dual_memory(
                        tails,
                        active_tail_exist,
                        anchor_memory,
                        anchor_support,
                        tdom,
                        scale_id,
                    )
                )
            if self.tail_aggregation == "direct_mean":
                pass
            elif (
                self.attention_direction == "anchor_to_tail"
                and self.tail_aggregation == "tailwise_topk"
            ):
                tail_memory, tail_support = self._tailwise_topk_update(
                    tails, active_tail_exist, anchor_memory, anchor_support, scale_id, tdom
                )
            elif self.attention_direction == "anchor_to_tail":
                tail_memory, tail_support = self._anchor_query_tail_update(
                    tails,
                    active_tail_exist,
                    anchor_memory,
                    anchor_support,
                    scale_id,
                    tdom,
                    tail_score_bias=selected_scores,
                    raw_tail_views=raw_tails,
                )
            else:
                tail_memory, tail_support = self._tail_memory(
                    tails, active_tail_exist, anchor_tokens, anchor_valid, scale_id, tdom
                )
            beta = self._effective_tail_beta(scale_id, tdom)
            alpha = self.alpha[scale_id].to(dtype=tdom.dtype, device=tdom.device)
            memory = self._merge_anchor_tail(anchor_memory, tail_memory, beta, conditioned_path=True)
            fused.append(self._fuse_tdom_memory(tdom, memory, scale_id))

            aux["anchor_memory"].append(anchor_memory)
            aux["tail_memory"].append(tail_memory)
            aux["anchor_support"].append(anchor_support)
            aux["tail_support"].append(tail_support)
            aux["alpha"].append(alpha.detach())
            aux["tail_beta"].append(beta.detach())
            aux["swin_r_alpha"].append(swin_alpha)
            aux["swin_r_delta_ratio"].append(delta_ratio)
            aux["latent_pairwise_cos"].append(latent_cos)
            aux["latent_weight_entropy"].append(latent_entropy)
            aux["latent_view_weight_mean"].append(weight_mean)
            aux["tail_local_context_alpha"].append(tail_context_alpha)
            aux["tail_local_context_delta_ratio"].append(tail_context_ratio)
            aux["tail_dual_memory_context_alpha"].append(tail_dual_memory_alpha)
            aux["tail_dual_memory_context_delta_ratio"].append(tail_dual_memory_ratio)

        # There is no TDOM-query attention map.  Returning None preserves the
        # existing EncoderDecoderCrossAttn calling convention without materialising
        # prohibitively large per-pixel view-attention tensors.
        if output_attn and return_aux:
            return fused, None, aux
        if output_attn:
            return fused, None
        if return_aux:
            return fused, aux
        return fused
