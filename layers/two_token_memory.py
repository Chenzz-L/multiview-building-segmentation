"""Read out paired anchor/TDOM tail updates without collapsing their queries.

The anchor representation and the frozen TDOM feature each query the same
tail set.  This module combines the two resulting residuals only after that
cross-attention step.  Its learned correction is zero at initialization, so
the enclosing SAIL layer starts with its existing anchor-query behaviour.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TwoTokenTailReadout(nn.Module):
    """Map a per-pixel pair of tail updates back to one anchor-memory update."""

    def __init__(self, embed_dim: int, hidden_ratio: float = 1.0):
        super().__init__()
        hidden_dim = max(int(embed_dim * hidden_ratio), 1)
        self.norm = nn.LayerNorm(2 * embed_dim)
        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        # Preserve the single-anchor-query path exactly at initialization.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, updates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [pixel, 1, channel] update and relative learned correction."""
        if updates.ndim != 3 or updates.shape[1] != 2:
            raise ValueError("Expected paired updates with shape [N, 2, C]")
        anchor_update = updates[:, 0]
        pair = torch.cat((updates[:, 0], updates[:, 1]), dim=-1)
        correction = self.fc2(self.act(self.fc1(self.norm(pair))))
        output = anchor_update + correction
        ratio = correction.float().norm(dim=-1).mean() / (
            anchor_update.float().norm(dim=-1).mean().clamp_min(1e-6)
        )
        return output.unsqueeze(1), ratio


class ThreeWayGatedTailReadout(nn.Module):
    """Select anchor, TDOM, or no tail update at every ground pixel.

    The two memory tokens remain independent through cross-attention.  This
    module makes the final decision explicit instead of freely adding an MLP
    correction to the anchor update:

    ``delta = g_anchor * delta_anchor + g_tdom * delta_tdom + g_none * 0``.

    The gate is deliberately initialized to reproduce the pre-existing
    single-anchor route: use the anchor update where anchor evidence exists;
    otherwise choose ``none``.  Thus the new decision path starts as a safe
    residual change and must be learned from the segmentation objective.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 32):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        # Scalar learned probes keep the gate cheap at res2 while letting it
        # inspect both query identities, both tail updates, and their
        # disagreements.  The gate itself is spatially varying, not global.
        self.probes = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(6)])
        self.norm = nn.LayerNorm(6)
        self.fc1 = nn.Linear(6, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 3)
        # Zero correction makes the fixed safe prior below exact at step 0.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        updates: torch.Tensor,
        queries: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the gated tail update and mean ``[anchor, tdom, none]`` gate.

        Args:
            updates: paired tail updates, ``[N, 2, C]``.
            queries: paired memory queries before tail attention, ``[N, 2, C]``.
            anchor_valid: whether the anchor query has a valid projected-image
                memory token, shape ``[N]``.
        """
        if updates.ndim != 3 or updates.shape[1] != 2:
            raise ValueError("Expected paired updates with shape [N, 2, C]")
        if queries.shape != updates.shape:
            raise ValueError("queries must have the same [N, 2, C] shape as updates")
        if anchor_valid.shape != (updates.shape[0],):
            raise ValueError("anchor_valid must have shape [N]")

        anchor_update, tdom_update = updates[:, 0], updates[:, 1]
        anchor_query, tdom_query = queries[:, 0], queries[:, 1]
        evidence = (
            anchor_update,
            tdom_update,
            anchor_query,
            tdom_query,
            (anchor_update - tdom_update).abs(),
            (anchor_query - tdom_query).abs(),
        )
        descriptor = torch.cat(
            [probe(feature) for probe, feature in zip(self.probes, evidence)], dim=-1
        )
        learned_logits = self.fc2(self.act(self.fc1(self.norm(descriptor))))

        # Safe baseline: existing valid-anchor pixels take delta_anchor;
        # anchor gaps take the explicit no-update route.  A large finite value
        # instead of -inf keeps mixed-precision softmax stable.
        base_logits = updates.new_empty((updates.shape[0], 3))
        base_logits[:] = torch.tensor(
            (-6.0, -6.0, 6.0), device=updates.device, dtype=updates.dtype
        )
        base_logits[anchor_valid] = torch.tensor(
            (6.0, -6.0, -6.0), device=updates.device, dtype=updates.dtype
        )
        weights = torch.softmax((base_logits + learned_logits).float(), dim=-1)
        weights = weights.to(dtype=updates.dtype)
        output = (
            weights[:, 0:1] * anchor_update
            + weights[:, 1:2] * tdom_update
        )
        return output.unsqueeze(1), weights.float().mean(dim=0)
