import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer


class GlobalSpatialSelfAttention(nn.Module):
    """Optional global spatial refinement used by legacy fusion variants."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                groups=embed_dim,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.self_attn = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_with_pos = x + self.pos_embed(x)
        x_seq = x_with_pos.flatten(2).transpose(1, 2)
        attended = self.self_attn(x_seq)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )
        return x + attended


class CrossAttentionBlock(nn.Module):
    """
    Fixed-region counterpart of PyTorchDeformableCrossAttentionBlock.

    The scoring, camera gate, residual gate, FFN, zero-candidate behavior,
    and optional evidence branch match deformable_fusion.py. The only major
    difference is sampling: this block receives a fixed local region.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_cameras=5,
        region_size=3,
        ffn_ratio=4.0,
        dropout=0.1,
        dot_weight=0.0,
        use_evidence_gate=False,
        evidence_score_weight=0.2,
        value_mode="projected",
        null_support_threshold=0.0,
        null_support_min_cameras=2,
        null_support_score_margin=0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if region_size <= 0 or region_size % 2 == 0:
            raise ValueError("region_size must be a positive odd integer")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_cameras = num_cameras
        self.region_size = region_size
        self.num_points = region_size ** 2
        self.head_dim = embed_dim // num_heads
        self.dot_weight = dot_weight
        self.use_evidence_gate = use_evidence_gate
        self.evidence_score_weight = evidence_score_weight
        self.null_support_threshold = null_support_threshold
        self.null_support_min_cameras = null_support_min_cameras
        self.null_support_score_margin = null_support_score_margin
        self.value_mode = value_mode
        if self.value_mode not in ("projected", "raw"):
            raise ValueError("value_mode must be 'projected' or 'raw'")

        self.norm_q = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = (
            nn.Identity()
            if self.value_mode == "raw"
            else nn.Linear(embed_dim, embed_dim)
        )
        self.output_proj = (
            nn.Identity()
            if self.value_mode == "raw"
            else nn.Linear(embed_dim, embed_dim)
        )

        self.lpe_proj = nn.Sequential(
            nn.Linear(8, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, self.head_dim),
        )
        self.lpe_scale = 0.1

        if self.use_evidence_gate:
            self.evidence_norm = nn.LayerNorm(self.head_dim)
            self.evidence_head = nn.Linear(self.head_dim, 1)
            nn.init.normal_(self.evidence_head.weight, std=0.02)
            nn.init.zeros_(self.evidence_head.bias)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        ffn_hidden = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim),
            nn.Dropout(dropout),
        )

        self.gate = nn.Parameter(torch.ones(1) * 10.0)
        self.register_buffer("tau", torch.tensor(0.5))

        radius = region_size // 2
        y_offset, x_offset = torch.meshgrid(
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            indexing="ij",
        )
        fixed_offsets = torch.stack(
            [x_offset.flatten(), y_offset.flatten()],
            dim=-1,
        )
        self.register_buffer("fixed_offsets", fixed_offsets)

    def forward(
        self,
        query,
        region_features,
        invalid=None,
        output_attn=False,
        return_aux=False,
    ):
        batch_size, num_queries, channels = query.shape
        _, _, num_cameras, num_points, _ = region_features.shape
        region_features = region_features.to(query.dtype)
        if num_cameras != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, got {num_cameras}"
            )
        if num_points != self.num_points:
            raise ValueError(
                f"Expected {self.num_points} region points, got {num_points}"
            )

        if invalid is None:
            invalid = torch.zeros(
                batch_size,
                num_queries,
                num_cameras,
                num_points,
                dtype=torch.bool,
                device=query.device,
            )

        # A zero image feature is an explicit empty/null candidate. It can
        # compete only when no nonzero candidates give cross-view support.
        zero_key_mask = ~region_features.ne(0).any(dim=-1)

        q_norm = self.norm_q(query)
        q = self.q_proj(q_norm).reshape(
            batch_size,
            num_queries,
            self.num_heads,
            self.head_dim,
        )

        k = self.k_proj(region_features).reshape(
            batch_size,
            num_queries,
            num_cameras,
            num_points,
            self.num_heads,
            self.head_dim,
        )
        k = k.masked_fill(
            zero_key_mask.unsqueeze(-1).unsqueeze(-1),
            0.0,
        )
        v = self.v_proj(region_features).reshape(
            batch_size,
            num_queries,
            num_cameras,
            num_points,
            self.num_heads,
            self.head_dim,
        )
        k = k.permute(0, 1, 4, 2, 5, 3)
        v = v.permute(0, 1, 4, 2, 5, 3)
        v = v.masked_fill(
            zero_key_mask.unsqueeze(2).unsqueeze(-2),
            0.0,
        )
        raw_k = k

        base_invalid = invalid
        zero_key_heads = (
            zero_key_mask.unsqueeze(2)
            .expand(-1, -1, self.num_heads, -1, -1)
        )
        k = k.masked_fill(
            zero_key_heads.unsqueeze(-2),
            0.0,
        )

        offsets = self.fixed_offsets.to(query.dtype).view(
            1,
            1,
            1,
            1,
            num_points,
            2,
        )
        lpe_features = torch.cat(
            [
                offsets,
                offsets ** 2,
                torch.sin(offsets),
                torch.cos(offsets),
            ],
            dim=-1,
        )
        lpe = self.lpe_proj(lpe_features)
        v = v * (
            1.0
            + self.lpe_scale
            * torch.clamp(lpe.transpose(-1, -2), -10, 10)
        )

        q_expanded = q.view(
            batch_size,
            num_queries,
            self.num_heads,
            1,
            self.head_dim,
            1,
        ).expand(-1, -1, -1, num_cameras, -1, num_points)
        dot_logits = (q_expanded * k).sum(dim=-2) / (
            self.head_dim ** 0.5
        )
        sim_logits = (
            F.normalize(q_expanded, dim=-2)
            * F.normalize(k, dim=-2)
        ).sum(dim=-2)

        # Zero image features are empty candidates and never participate in
        # attention. If every candidate is zero/invalid, the image fusion output
        # for that query is forced to zero after softmax.
        invalid_heads = base_invalid.unsqueeze(2).expand(
            -1,
            -1,
            self.num_heads,
            -1,
            -1,
        ) | zero_key_heads
        evidence_logits = None
        evidence_gate = None
        best_evidence = None
        all_invalid = invalid_heads.reshape(
            batch_size,
            num_queries,
            self.num_heads,
            -1,
        ).all(dim=-1)
        no_value = (
            base_invalid.unsqueeze(2) | zero_key_heads
        ).reshape(
            batch_size,
            num_queries,
            self.num_heads,
            -1,
        ).all(dim=-1)

        if self.use_evidence_gate:
            evidence_features = raw_k.permute(0, 1, 2, 3, 5, 4)
            evidence_logits = self.evidence_head(
                self.evidence_norm(evidence_features)
            ).squeeze(-1)
            evidence_logits = evidence_logits.masked_fill(
                invalid_heads,
                -1e4,
            )
            best_evidence = evidence_logits.reshape(
                batch_size,
                num_queries,
                self.num_heads,
                -1,
            ).max(dim=-1).values
            best_evidence = best_evidence.masked_fill(all_invalid, -20.0)
            evidence_gate = torch.sigmoid(best_evidence)

        attn_logits = sim_logits
        if evidence_logits is not None:
            attn_logits = (
                attn_logits
                + self.evidence_score_weight * evidence_logits
            )
        attn_logits = attn_logits / self.tau
        attn_logits = attn_logits.masked_fill(invalid_heads, -1e9)

        attn_weights = F.softmax(
            attn_logits.reshape(
                batch_size,
                num_queries,
                self.num_heads,
                -1,
            ),
            dim=-1,
        ).reshape(
            batch_size,
            num_queries,
            self.num_heads,
            num_cameras,
            num_points,
        )
        attn_weights = attn_weights.masked_fill(
            all_invalid.reshape(
                batch_size,
                num_queries,
                self.num_heads,
                1,
                1,
            ),
            0.0,
        )

        v_flat = v.permute(0, 1, 2, 3, 5, 4)
        out = (
            attn_weights.reshape(
                batch_size,
                num_queries,
                self.num_heads,
                1,
                -1,
            )
            @ v_flat.reshape(
                batch_size,
                num_queries,
                self.num_heads,
                -1,
                self.head_dim,
            )
        ).squeeze(-2)
        out = out.masked_fill(no_value.unsqueeze(-1), 0.0)
        out = out.reshape(batch_size, num_queries, self.embed_dim)
        x = self.output_proj(out)
        pixel_no_value = no_value.all(dim=-1)
        x = x.masked_fill(pixel_no_value.unsqueeze(-1), 0.0)
        if evidence_gate is not None:
            x = x * evidence_gate.mean(dim=2).unsqueeze(-1)

        gate_scale = torch.sigmoid(self.gate - 3.5)
        result = q_norm + gate_scale * x
        result = result + self.ffn(self.norm_ffn(result))

        if output_attn or return_aux:
            extras = {}
            if output_attn:
                extras["attn"] = attn_weights.sum(dim=-1).detach()
                camera_invalid = invalid_heads.all(dim=-1)
                camera_cosine = sim_logits.masked_fill(
                    invalid_heads,
                    float("-inf"),
                ).max(dim=-1).values
                camera_cosine = camera_cosine.masked_fill(
                    camera_invalid,
                    float("nan"),
                )
                extras["cosine"] = camera_cosine.detach()
            if return_aux and evidence_logits is not None:
                extras["evidence_logits"] = best_evidence
                extras["evidence_gate"] = evidence_gate
                extras["evidence_valid"] = ~all_invalid
            return result, extras
        return result


class MultiScaleCrossAttentionFusion(nn.Module):
    """Multi-scale fixed-region cross-attention fusion."""

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_cameras=5,
        num_scales=4,
        ffn_ratio=4.0,
        dropout=0.1,
        region_size=3,
        dot_weight=0.0,
        use_evidence_gate=False,
        evidence_score_weight=0.2,
        value_mode="projected",
        null_support_threshold=0.0,
        null_support_min_cameras=2,
        null_support_score_margin=0.1,
        use_global_attn=False,
        global_attn_scales=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.num_scales = num_scales
        self.region_size = region_size
        self.num_points = region_size ** 2
        self.scale_keys = ["res2", "res3", "res4", "res5"]

        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_cameras=num_cameras,
                    region_size=region_size,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    dot_weight=dot_weight,
                    use_evidence_gate=use_evidence_gate,
                    evidence_score_weight=evidence_score_weight,
                    value_mode=value_mode,
                    null_support_threshold=null_support_threshold,
                    null_support_min_cameras=null_support_min_cameras,
                    null_support_score_margin=null_support_score_margin,
                )
                for _ in range(num_scales)
            ]
        )
        self.scale_embed = nn.Embedding(num_scales, embed_dim)

        self.use_global_attn = use_global_attn
        self.global_attn_scales = (
            [2, 3] if global_attn_scales is None else global_attn_scales
        )
        if self.use_global_attn:
            self.global_attn_blocks = nn.ModuleDict(
                {
                    str(scale_idx): GlobalSpatialSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                    for scale_idx in self.global_attn_scales
                }
            )

    def forward(
        self,
        tdom_features,
        cam_features,
        cam_exist=None,
        output_attn=False,
        return_aux=False,
    ):
        fused_features = []
        all_attns = {}
        auxiliary = {}

        for scale_idx, scale_key in enumerate(self.scale_keys):
            tdom_feature = tdom_features[scale_idx]
            batch_size, channels, height, width = tdom_feature.shape
            query = tdom_feature.flatten(2).transpose(1, 2)
            query = query + self.scale_embed.weight[scale_idx].view(
                1,
                1,
                channels,
            )

            camera_regions = []
            for camera_idx in range(self.num_cameras):
                camera_feature = cam_features[camera_idx][scale_key]
                if camera_feature.shape[-2:] != (height, width):
                    camera_feature = F.interpolate(
                        camera_feature,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )

                unfolded = F.unfold(
                    camera_feature,
                    kernel_size=self.region_size,
                    padding=self.region_size // 2,
                )
                region = unfolded.view(
                    batch_size,
                    channels,
                    self.num_points,
                    height * width,
                ).permute(0, 3, 2, 1)
                camera_regions.append(region)

            region_features = torch.stack(camera_regions, dim=2)
            invalid = torch.zeros(
                batch_size,
                height * width,
                self.num_cameras,
                self.num_points,
                dtype=torch.bool,
                device=tdom_feature.device,
            )
            if cam_exist is not None:
                camera_missing = ~cam_exist.bool()
                invalid = invalid | camera_missing.view(
                    batch_size,
                    1,
                    self.num_cameras,
                    1,
                )

            if output_attn or return_aux:
                out, extras = self.cross_attn_blocks[scale_idx](
                    query,
                    region_features,
                    invalid,
                    output_attn=output_attn,
                    return_aux=return_aux,
                )
                if output_attn:
                    all_attns[scale_key] = extras
                if return_aux:
                    for aux_key in (
                        "evidence_logits",
                        "evidence_gate",
                        "evidence_valid",
                    ):
                        aux_value = extras.get(aux_key)
                        if aux_value is not None:
                            extras[aux_key] = aux_value.permute(
                                0,
                                2,
                                1,
                            ).reshape(
                                batch_size,
                                -1,
                                height,
                                width,
                            )
                    auxiliary[scale_key] = extras
            else:
                out = self.cross_attn_blocks[scale_idx](
                    query,
                    region_features,
                    invalid,
                )

            out = out.transpose(1, 2).reshape(
                batch_size,
                channels,
                height,
                width,
            )
            if (
                self.use_global_attn
                and scale_idx in self.global_attn_scales
            ):
                out = self.global_attn_blocks[str(scale_idx)](out)
            fused_features.append(out)

        if output_attn and return_aux:
            return fused_features, all_attns, auxiliary
        if output_attn:
            return fused_features, all_attns
        if return_aux:
            return fused_features, auxiliary
        return fused_features
