"""K-to-M latent spatial memory construction for SAIL anchors.

The module keeps a small set of permutation-friendly latent view-composition
maps instead of immediately collapsing K anchor maps to one mean.  Each
latent map receives shared resolution-preserving Swin refinement; only the M
latent tokens at the same spatial position interact afterwards.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from layers.swin_r import SwinRRefiner


class MultiLatentSwinR(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_latents: int = 3,
        num_heads: int = 8,
        spatial_window_size: int = 8,
        spatial_depth: int = 2,
        latent_depth: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        beta_init: float = 0.01,
        alpha_init: float = 0.01,
    ):
        super().__init__()
        if num_latents < 1:
            raise ValueError("num_latents must be >= 1")
        if latent_depth < 1:
            raise ValueError("latent_depth must be >= 1")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_latents = int(num_latents)
        self.embed_dim = int(embed_dim)

        # M distinct learned queries read the unordered K-anchor set.  In
        # contrast with the previous zero-initialised M-output scorer, the
        # queries deliberately break latent-axis symmetry from step zero
        # while retaining permutation equivariance over anchor views.
        self.composition_norm = nn.LayerNorm(embed_dim)
        self.latent_queries = nn.Parameter(torch.empty(num_latents, embed_dim))
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        # Independent residual amplitudes keep the stable R0 path while
        # allowing each query-pooled latent to depart at its own rate.
        self.beta = nn.Parameter(torch.full((num_latents,), float(beta_init)))

        # The same spatial refiner is applied to every latent map by folding M
        # into the batch dimension.
        self.spatial = SwinRRefiner(
            embed_dim=embed_dim,
            window_size=spatial_window_size,
            num_heads=num_heads,
            depth=spatial_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            alpha_init=alpha_init,
        )

        self.latent_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(latent_depth)])
        self.latent_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(latent_depth)
        ])
        self.latent_ffn_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(latent_depth)])
        hidden = int(embed_dim * mlp_ratio)
        self.latent_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
            ) for _ in range(latent_depth)
        ])
        # Start as an exact late-pooling mean; the latent-axis branch can
        # learn to exchange information without perturbing the baseline on
        # the first optimization step.
        self.latent_residual_scale = nn.Parameter(torch.tensor(0.0))

        # Learned late pooling is deliberately after spatial and latent-axis
        # interaction.  Zero init starts as a uniform latent mean.
        self.pool_norm = nn.LayerNorm(embed_dim)
        self.pool_logits = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.pool_logits.weight)
        nn.init.zeros_(self.pool_logits.bias)

    @staticmethod
    def _masked_mean(views: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = valid.unsqueeze(2).to(dtype=views.dtype)
        count = weights.sum(dim=1)
        mean = (views * weights).sum(dim=1) / count.clamp_min(1.0)
        mean = torch.where(count > 0, mean, torch.zeros_like(mean))
        return mean, count

    @staticmethod
    def _safe_masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # logits [B,M,K,H,W], valid [B,K,H,W].
        mask = valid.unsqueeze(1)
        logits = logits.masked_fill(~mask, -float("inf"))
        all_invalid = ~valid.any(dim=1)
        if all_invalid.any():
            logits = logits.clone()
            first = logits[:, :, 0]
            first = torch.where(
                all_invalid[:, None], torch.zeros_like(first), first
            )
            logits[:, :, 0] = first
        weights = torch.softmax(logits.float(), dim=2).to(dtype=logits.dtype)
        weights = weights * mask.to(dtype=weights.dtype)
        return weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _safe_key_padding_mask(valid: torch.Tensor):
        # valid [N,M], avoiding all-masked MHA rows.
        padding = ~valid
        empty = ~valid.any(dim=1)
        if empty.any():
            padding = padding.clone()
            padding[empty, 0] = False
        return padding, empty

    def _latent_axis_attention(self, latent: torch.Tensor, valid_map: torch.Tensor) -> torch.Tensor:
        b, m, c, h, w = latent.shape
        tokens = latent.permute(0, 3, 4, 1, 2).reshape(b * h * w, m, c)
        valid = valid_map[:, None].expand(b, m, h, w).permute(0, 2, 3, 1).reshape(b * h * w, m)
        for norm, attn, ffn_norm, ffn in zip(
            self.latent_norm, self.latent_attn, self.latent_ffn_norm, self.latent_ffn
        ):
            key_padding, empty = self._safe_key_padding_mask(valid)
            attended, _ = attn(norm(tokens), norm(tokens), norm(tokens),
                               key_padding_mask=key_padding, need_weights=False)
            candidate = tokens + attended
            delta = attended + ffn(ffn_norm(candidate))
            tokens = tokens + self.latent_residual_scale.to(dtype=tokens.dtype) * delta
            tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
            if empty.any():
                tokens[empty] = 0.0
        return tokens.reshape(b, h, w, m, c).permute(0, 3, 4, 1, 2)

    @staticmethod
    def _pairwise_cosine(latent: torch.Tensor, valid_map: torch.Tensor) -> torch.Tensor:
        b, m, c, h, w = latent.shape
        tokens = latent.permute(0, 1, 3, 4, 2).reshape(b, m, h * w, c)
        normed = torch.nn.functional.normalize(tokens.float(), dim=-1, eps=1e-6)
        cosine = torch.einsum("bmnc,bknc->bmkn", normed, normed)
        mask = valid_map.reshape(b, 1, 1, h * w).float()
        cosine = cosine * mask
        if m > 1:
            off_diag = ~torch.eye(m, device=latent.device, dtype=torch.bool)[None, :, :, None]
            values = cosine[off_diag.expand(b, -1, -1, h * w)]
            return values.mean() if values.numel() else latent.new_zeros(())
        return latent.new_zeros(())

    def forward(
        self, anchors: torch.Tensor, valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pooled memory, support, latent maps, latent validity, weights and stats."""
        b, k, c, h, w = anchors.shape
        if c != self.embed_dim or valid.shape != (b, k, h, w):
            raise ValueError("Invalid anchor/valid shapes for MultiLatentSwinR")
        mean, count = self._masked_mean(anchors, valid)
        per_view = anchors.permute(0, 1, 3, 4, 2).contiguous()  # [B,K,H,W,C]
        keys = self.key_proj(self.composition_norm(per_view))
        queries = self.query_proj(self.latent_queries)
        logits = torch.einsum("mc,bkhwc->bmkhw", queries, keys)
        logits = logits * (self.embed_dim ** -0.5)
        weights = self._safe_masked_softmax(logits, valid)
        composed = torch.einsum("bmkhw,bkchw->bmchw", weights, anchors)
        beta = self.beta.to(dtype=anchors.dtype).view(1, self.num_latents, 1, 1, 1)
        latent = mean.unsqueeze(1) + beta * (composed - mean.unsqueeze(1))
        latent_before_swin = latent

        valid_map = count[:, 0] > 0
        flat = latent.reshape(b * self.num_latents, c, h, w)
        flat_valid = valid_map[:, None].expand(b, self.num_latents, h, w).reshape(
            b * self.num_latents, h, w
        )
        refined, spatial_alpha, delta_ratio = self.spatial(flat, flat_valid)
        latent = refined.reshape(b, self.num_latents, c, h, w)
        latent_after_swin = latent
        latent = self._latent_axis_attention(latent, valid_map)

        # Read-only diagnostics for validation hooks.  Detaching ensures this
        # instrumentation never changes the training graph or loss.
        self.last_diagnostics = {
            "r0": mean.detach(),
            "rhat": composed.detach(),
            "latent_before_swin": latent_before_swin.detach(),
            "latent_after_swin": latent_after_swin.detach(),
            "latent_after_axis": latent.detach(),
            "valid": valid_map.detach(),
            "weights": weights.detach(),
        }

        pool_tokens = latent.permute(0, 3, 4, 1, 2).reshape(b * h * w, self.num_latents, c)
        pool_scores = self.pool_logits(self.pool_norm(pool_tokens)).squeeze(-1)
        latent_valid = valid_map.reshape(b * h * w, 1).expand(-1, self.num_latents)
        pool_scores = pool_scores.masked_fill(~latent_valid, -float("inf"))
        empty = ~latent_valid.any(dim=1)
        if empty.any():
            pool_scores = pool_scores.clone()
            pool_scores[empty, 0] = 0.0
        pool_weights = torch.softmax(pool_scores.float(), dim=1).to(dtype=latent.dtype)
        pool_weights = pool_weights * latent_valid.to(dtype=pool_weights.dtype)
        pooled = torch.einsum("nm,nmc->nc", pool_weights, pool_tokens)
        pooled = pooled.reshape(b, h, w, c).permute(0, 3, 1, 2)
        pooled = pooled.masked_fill(~valid_map.unsqueeze(1), 0.0)
        entropy = -(pool_weights.clamp_min(1e-8) * pool_weights.clamp_min(1e-8).log()).sum(dim=1).mean()
        pairwise = self._pairwise_cosine(latent, valid_map)
        weight_mean = weights.float().mean(dim=(0, 3, 4)).detach()
        latent_support = valid_map[:, None].expand(b, self.num_latents, h, w)
        return (
            pooled,
            count / float(max(k, 1)),
            latent,
            latent_support,
            weights,
            spatial_alpha,
            delta_ratio,
            torch.stack((pairwise, entropy)),
            weight_mean,
        )
