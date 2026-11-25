import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateGateFusionBlock(nn.Module):
    """
    Build per-view fused candidates first, then choose among them.

    For each pixel and scale:
        candidate_k = TDOM + image_feature_k
        score_k = shared_score_net(candidate_k)
        fused = TDOM + sum_k softmax(score_k) * image_feature_k

    Zero image features can participate only when no nonzero candidate exists;
    because their value is zero, they act as a no-image fallback.
    """

    def __init__(
        self,
        embed_dim=256,
        num_cameras=5,
        hidden_ratio=0.25,
        dropout=0.1,
        null_support_min_cameras=2,
        null_support_score_margin=0.1,
    ):
        super().__init__()
        hidden_dim = max(16, int(embed_dim * hidden_ratio))
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.null_support_min_cameras = null_support_min_cameras
        self.null_support_score_margin = null_support_score_margin

        self.candidate_norm = nn.GroupNorm(1, embed_dim)
        self.candidate_ffn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.Dropout2d(dropout),
        )
        self.score_norm = nn.GroupNorm(1, embed_dim)
        self.score_net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.output_norm = nn.GroupNorm(1, embed_dim)
        self.output_ffn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.Dropout2d(dropout),
        )
        self.gate = nn.Parameter(torch.ones(1) * 10.0)
        self.last_debug_stats = {}
        nn.init.zeros_(self.candidate_ffn[-2].weight)
        nn.init.zeros_(self.candidate_ffn[-2].bias)
        nn.init.zeros_(self.score_net[-1].weight)
        nn.init.zeros_(self.score_net[-1].bias)
        nn.init.zeros_(self.output_ffn[-2].weight)
        nn.init.zeros_(self.output_ffn[-2].bias)

    def forward(
        self,
        query,
        cam_stack,
        cam_exist=None,
        output_attn=False,
        return_aux=False,
    ):
        B, C, H, W = query.shape
        _, K, _, _, _ = cam_stack.shape
        cam_stack = cam_stack.to(query.dtype)

        raw_valid = cam_stack.detach().abs().sum(dim=2) > 1e-6
        if cam_exist is not None:
            cam_valid = cam_exist.bool().view(B, K, 1, 1)
            raw_valid = raw_valid & cam_valid

        candidates = query.unsqueeze(1) + cam_stack
        candidates_flat = candidates.reshape(B * K, C, H, W)
        processed_flat = candidates_flat + self.candidate_ffn(
            self.candidate_norm(candidates_flat)
        )
        processed_candidates = processed_flat.reshape(B, K, C, H, W)

        scores = self.score_net(self.score_norm(processed_flat))
        scores = scores.reshape(B, K, H, W)

        nonzero_scores = scores.masked_fill(~raw_valid, float("-inf"))
        top_k = min(max(self.null_support_min_cameras, 1), K)
        top_scores = nonzero_scores.topk(top_k, dim=1).values
        kth_score = top_scores[:, -1]
        score_spread = top_scores[:, 0] - kth_score
        has_support = (
            torch.isfinite(kth_score)
            & (score_spread <= self.null_support_score_margin)
        )

        # Zero candidates are kept only when there is no multi-view support.
        valid_for_softmax = raw_valid | (~has_support.unsqueeze(1))
        masked_scores = scores.masked_fill(~valid_for_softmax, float("-inf"))
        all_invalid = ~torch.isfinite(masked_scores).any(dim=1, keepdim=True)
        masked_scores = masked_scores.masked_fill(all_invalid, 0.0)
        weights = F.softmax(masked_scores, dim=1)
        weights = weights.masked_fill(all_invalid, 0.0)

        candidate_delta = processed_candidates - query.unsqueeze(1)
        img_delta = (weights.unsqueeze(2) * candidate_delta).sum(dim=1)
        gate_scale = torch.sigmoid(self.gate - 3.5)
        fused = query + gate_scale * img_delta
        fused = fused + self.output_ffn(self.output_norm(fused))

        with torch.no_grad():
            valid_queries = ~all_invalid.squeeze(1)
            if valid_queries.any():
                w_flat = weights.permute(0, 2, 3, 1)[valid_queries]
                entropy = -(w_flat * (w_flat + 1e-8).log()).sum(dim=-1).mean()
                max_attn = w_flat.max(dim=-1).values.mean()
                valid_ratio = raw_valid.float().mean()
            else:
                entropy = torch.tensor(0.0, device=query.device)
                max_attn = torch.tensor(0.0, device=query.device)
                valid_ratio = torch.tensor(0.0, device=query.device)
            self.last_debug_stats = {
                "entropy": float(entropy.detach().cpu()),
                "max_attn": float(max_attn.detach().cpu()),
                "offset_mean": 0.0,
                "valid_ratio": float(valid_ratio.detach().cpu()),
            }

        if output_attn or return_aux:
            extras = {}
            if output_attn:
                attn = weights.flatten(2).transpose(1, 2).unsqueeze(2)
                extras["attn"] = attn.detach()
                extras["score"] = (
                    scores.flatten(2).transpose(1, 2).unsqueeze(2).detach()
                )
            return fused, extras
        return fused


class MultiScaleCandidateGateFusion(nn.Module):
    """Multi-scale candidate-level gate fusion."""

    def __init__(
        self,
        embed_dim=256,
        num_cameras=5,
        num_scales=4,
        dropout=0.1,
        hidden_ratio=0.25,
        null_support_min_cameras=2,
        null_support_score_margin=0.1,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_cameras = num_cameras
        self.scale_keys = ["res2", "res3", "res4", "res5"]
        self.blocks = nn.ModuleList(
            [
                CandidateGateFusionBlock(
                    embed_dim=embed_dim,
                    num_cameras=num_cameras,
                    hidden_ratio=hidden_ratio,
                    dropout=dropout,
                    null_support_min_cameras=null_support_min_cameras,
                    null_support_score_margin=null_support_score_margin,
                )
                for _ in range(num_scales)
            ]
        )

    def forward(
        self,
        tdom_features,
        cam_features,
        cam_exist=None,
        output_attn=False,
        return_aux=False,
    ):
        fused_features, attns, auxiliary = [], {}, {}
        for i, scale_key in enumerate(self.scale_keys[: self.num_scales]):
            query = tdom_features[i]
            cam_list = []
            for cam_idx in range(self.num_cameras):
                cf = cam_features[cam_idx][scale_key]
                if cf.shape[-2:] != query.shape[-2:]:
                    cf = F.interpolate(
                        cf,
                        size=query.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                cam_list.append(cf)
            cam_stack = torch.stack(cam_list, dim=1)

            if output_attn or return_aux:
                out, extras = self.blocks[i](
                    query,
                    cam_stack,
                    cam_exist=cam_exist,
                    output_attn=output_attn,
                    return_aux=return_aux,
                )
                if output_attn:
                    attns[scale_key] = extras
                if return_aux:
                    auxiliary[scale_key] = extras
            else:
                out = self.blocks[i](
                    query,
                    cam_stack,
                    cam_exist=cam_exist,
                )
            fused_features.append(out)

        if output_attn and return_aux:
            return fused_features, attns, auxiliary
        if output_attn:
            return fused_features, attns
        if return_aux:
            return fused_features, auxiliary
        return fused_features
