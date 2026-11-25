import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchDeformableCrossAttentionBlock(nn.Module):
    """
    Geometry + Probabilistic Deformable Cross Attention
    """
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_cameras=5,
        num_points=4,
        ffn_ratio=4.0,
        dropout=0.1,
        max_offset=2.5,
        dot_weight=0.0,
        use_evidence_gate=False,
        evidence_score_weight=0.2,
        value_mode="projected",
        fusion_mode="standard",
        null_support_threshold=0.0,
        null_support_min_cameras=2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_cameras = num_cameras
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.max_offset = max_offset
        self.dot_weight = dot_weight
        self.use_evidence_gate = use_evidence_gate
        self.evidence_score_weight = evidence_score_weight
        self.null_support_threshold = null_support_threshold
        self.null_support_min_cameras = null_support_min_cameras
        self.value_mode = value_mode
        if self.value_mode not in ("projected", "raw"):
            raise ValueError("value_mode must be 'projected' or 'raw'")
        self.fusion_mode = fusion_mode
        if self.fusion_mode not in ("standard", "raw_add"):
            raise ValueError("fusion_mode must be 'standard' or 'raw_add'")
        if self.fusion_mode == "raw_add" and self.value_mode != "raw":
            raise ValueError("fusion_mode='raw_add' requires value_mode='raw'")

        self.norm_q = nn.LayerNorm(embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_cameras * num_points * 2)

        self.lpe_proj = None
        if self.fusion_mode == "standard":
            self.lpe_proj = nn.Sequential(
                nn.Linear(8, self.head_dim),
                nn.GELU(),
                nn.Linear(self.head_dim, self.head_dim)
            )
        self.lpe_scale = 0.1

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = (
            nn.Identity()
            if self.value_mode == "raw"
            else nn.Linear(embed_dim, embed_dim)
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = (
            nn.Identity()
            if self.value_mode == "raw"
            else nn.Linear(embed_dim, embed_dim)
        )
        if self.use_evidence_gate:
            self.evidence_norm = nn.LayerNorm(self.head_dim)
            self.evidence_head = nn.Linear(self.head_dim, 1)
            nn.init.normal_(self.evidence_head.weight, std=0.02)
            nn.init.zeros_(self.evidence_head.bias)
        self.norm_ffn = None
        self.ffn = None
        self.gate = None
        if self.fusion_mode == "standard":
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
        self.register_buffer("tau", torch.tensor(0.5)) # fixed attention temperature
        self.last_debug_stats = {}

        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.uniform_(self.sampling_offsets.bias, -0.1, 0.1)

    def forward(
        self,
        query,
        cam_features,
        cam_exist=None,
        output_attn=False,
        return_aux=False,
        residual_query=None,
    ):
        B, C, Hq, Wq = query.shape
        _, N_cam, _, H, W = cam_features.shape
        Lq = Hq * Wq
        cam_features = cam_features.to(query.dtype)

        # 1. Projections
        query_hwc = query.permute(0, 2, 3, 1)
        q_norm = self.norm_q(query_hwc).view(B, Lq, C)
        q = self.q_proj(q_norm).reshape(B, Lq, self.num_heads, self.head_dim)

        # 2. offsets
        raw_offsets = self.sampling_offsets(q_norm).view(B, Lq, self.num_heads, N_cam, self.num_points, 2)
        offsets = torch.tanh(raw_offsets) * self.max_offset

        # 3. reference points
        y_grid, x_grid = torch.meshgrid(
            torch.arange(Hq, device=query.device),
            torch.arange(Wq, device=query.device),
            indexing='ij'
        )
        x_norm = (x_grid.float() / max(Wq - 1, 1)) * 2 - 1
        y_norm = (y_grid.float() / max(Hq - 1, 1)) * 2 - 1
        reference_points = torch.stack([x_norm, y_norm], dim=-1).view(1, Lq, 1, 1, 1, 2)
        offset_norm = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)], device=query.device).view(1, 1, 1, 1, 1, 2)
        sampling_locations = reference_points + offsets * offset_norm

        # 5. key/value projection
        kv_all = cam_features.view(B * N_cam, C, H, W).permute(0, 2, 3, 1) 
        k_all = self.k_proj(kv_all).view(B * N_cam, H, W, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)
        if self.value_mode == "raw":
            v_all = (
                cam_features.view(B * N_cam, C, H, W)
                .view(B * N_cam, self.num_heads, self.head_dim, H, W)
            )
        else:
            v_all = self.v_proj(kv_all).view(B * N_cam, H, W, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)
        v_all = v_all.reshape(B * N_cam * self.num_heads, self.head_dim, H, W)
        k_all = k_all.reshape(B * N_cam * self.num_heads, self.head_dim, H, W)

        # 6. sampling
        grid = sampling_locations.permute(0, 3, 2, 1, 4, 5).reshape(B * N_cam * self.num_heads, Lq, self.num_points, 2)
        raw_valid = (
            cam_features.detach().abs().sum(dim=2, keepdim=True) > 1e-6
        ).float()
        raw_valid = (
            raw_valid.view(B * N_cam, 1, H, W)
            .repeat_interleave(self.num_heads, dim=0)
        )
        valid_sampled = F.grid_sample(
            raw_valid,
            grid.float(),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        v_sampled = F.grid_sample(v_all.float(), grid.float(), mode='bilinear', padding_mode='zeros', align_corners=True).to(query.dtype)
        k_sampled = F.grid_sample(k_all.float(), grid.float(), mode='bilinear', padding_mode='zeros', align_corners=True).to(query.dtype)
        
        # 8. reshape sampled feat
        valid_sampled = valid_sampled.view(B, N_cam, self.num_heads, 1, Lq, self.num_points).permute(0, 4, 2, 1, 3, 5).squeeze(-2)
        v_sampled = v_sampled.view(B, N_cam, self.num_heads, self.head_dim, Lq, self.num_points).permute(0, 4, 2, 1, 3, 5)
        k_sampled = k_sampled.view(B, N_cam, self.num_heads, self.head_dim, Lq, self.num_points).permute(0, 4, 2, 1, 3, 5)
        empty_value = valid_sampled <= 1e-6
        v_sampled = v_sampled.masked_fill(empty_value.unsqueeze(-2), 0.0)
        k_sampled = k_sampled.masked_fill(empty_value.unsqueeze(-2), 0.0)

        # 9. valid mask: empty image features are treated as no evidence
        # in both training and evaluation for this experiment.
        valid_mask = torch.ones(B, Lq, self.num_heads, N_cam, self.num_points, dtype=torch.bool, device=query.device)
        if cam_exist is not None:
            cam_mask = cam_exist.bool().view(B, 1, 1, N_cam, 1)
            invalid = ~cam_mask.expand_as(valid_mask)
        else:
            invalid = torch.zeros_like(valid_mask)
        interaction_invalid = invalid | empty_value
        invalid = invalid | empty_value

        # 10. Independent candidate keys. Do not let candidates at the same
        # pixel alter each other's matching vectors before attention scoring.
        k_sampled_inter = k_sampled.masked_fill(
            interaction_invalid.unsqueeze(-2),
            0.0,
        )
        
        # 11. Value path. raw_add keeps sampled image features unchanged.
        if self.fusion_mode == "raw_add":
            v = v_sampled
        else:
            lpe_feat = torch.cat([offsets, offsets**2, torch.sin(offsets), torch.cos(offsets)], dim=-1)
            lpe = self.lpe_proj(lpe_feat)
            v = v_sampled * (1.0 + self.lpe_scale * torch.clamp(lpe.transpose(-1, -2), -10, 10))
        
        # 12. attention scoring
        q_exp = q.view(B, Lq, self.num_heads, 1, self.head_dim, 1).expand(-1, -1, -1, N_cam, -1, self.num_points)
        dot_logits = (q_exp * k_sampled_inter).sum(dim=-2) / (self.head_dim ** 0.5)

        q_normed = F.normalize(q_exp, dim=-2)
        k_normed = F.normalize(k_sampled_inter, dim=-2)
        sim_logits = (q_normed * k_normed).sum(dim=-2) 

        evidence_logits = None
        evidence_gate = None
        if self.use_evidence_gate:
            evidence_features = k_sampled.permute(0, 1, 2, 3, 5, 4)
            evidence_logits = self.evidence_head(
                self.evidence_norm(evidence_features)
            ).squeeze(-1)
            evidence_logits = evidence_logits.masked_fill(invalid, -1e4)

            best_evidence = evidence_logits.reshape(
                B, Lq, self.num_heads, -1
            ).max(dim=-1).values
            all_invalid = invalid.reshape(
                B, Lq, self.num_heads, -1
            ).all(dim=-1)
            best_evidence = best_evidence.masked_fill(all_invalid, -20.0)
            evidence_gate = torch.sigmoid(best_evidence)

        attn_logits = sim_logits + self.dot_weight * dot_logits
        if evidence_logits is not None:
            attn_logits = (
                attn_logits
                + self.evidence_score_weight * evidence_logits
            )
        attn_logits = attn_logits / self.tau
        attn_logits = attn_logits.masked_fill(invalid, -1e9)

        # 13. Softmax over real image sampling points.
        attn_weights = F.softmax(attn_logits.reshape(B, Lq, self.num_heads, -1), dim=-1).reshape(B, Lq, self.num_heads, N_cam, self.num_points)
        all_invalid = invalid.reshape(B, Lq, self.num_heads, -1).all(dim=-1)
        no_value = interaction_invalid.reshape(
            B,
            Lq,
            self.num_heads,
            -1,
        ).all(dim=-1)
        attn_weights = attn_weights.masked_fill(all_invalid.reshape(B, Lq, self.num_heads, 1, 1), 0.0)
        with torch.no_grad():
            w_flat_debug = attn_weights.reshape(B, Lq, self.num_heads, -1)
            valid_queries_debug = ~all_invalid
            if valid_queries_debug.any():
                valid_w_debug = w_flat_debug[valid_queries_debug]
                entropy_debug = -(
                    valid_w_debug * (valid_w_debug + 1e-8).log()
                ).sum(dim=-1).mean()
                max_attn_debug = valid_w_debug.max(dim=-1).values.mean()
                valid_candidate_ratio = (~invalid).to(torch.float32).mean()
                offset_mean_debug = offsets[~invalid].norm(
                    p=2,
                    dim=-1,
                ).mean() if (~invalid).any() else torch.tensor(
                    0.0,
                    device=query.device,
                )
            else:
                entropy_debug = torch.tensor(0.0, device=query.device)
                max_attn_debug = torch.tensor(0.0, device=query.device)
                valid_candidate_ratio = torch.tensor(0.0, device=query.device)
                offset_mean_debug = torch.tensor(0.0, device=query.device)
            self.last_debug_stats = {
                "entropy": float(entropy_debug.detach().cpu()),
                "max_attn": float(max_attn_debug.detach().cpu()),
                "offset_mean": float(offset_mean_debug.detach().cpu()),
                "valid_ratio": float(valid_candidate_ratio.detach().cpu()),
            }

        sim_flat = sim_logits.masked_fill(invalid, float("-inf")).reshape(
            B, Lq, self.num_heads, -1
        )
        best_sim_all = sim_flat.max(dim=-1).values

        # 14. monitoring
        if self.training and torch.rand(1).item() < 0.005:
            with torch.no_grad():
                v_mask_bool = ~invalid
                if v_mask_bool.any():
                    off_dist = offsets[v_mask_bool].norm(p=2, dim=-1).mean()
                    sim_std = sim_logits[v_mask_bool].std()
                    dot_std = dot_logits[v_mask_bool].std()
                else:
                    off_dist, sim_std, dot_std = [torch.tensor(0.0, device=query.device)] * 3

                w_flat = attn_weights.reshape(B, Lq, self.num_heads, -1)
                if (~all_invalid).any():
                    valid_w = w_flat[~all_invalid]
                    entropy = -(valid_w * (valid_w + 1e-8).log()).sum(dim=-1).mean()
                    max_attn = w_flat.max(dim=-1)[0][~all_invalid].mean()

                    best_sim = best_sim_all[~all_invalid].float()
                    best_sim_mean = best_sim.mean()
                    best_sim_std = best_sim.std()
                    best_sim_min = best_sim.min()
                    best_sim_max = best_sim.max()
                    best_sim_q = torch.quantile(
                        best_sim,
                        torch.tensor([0.1, 0.5, 0.9], device=best_sim.device),
                    )
                else:
                    entropy, max_attn = [torch.tensor(0.0, device=query.device)] * 2
                    best_sim_mean, best_sim_std, best_sim_min, best_sim_max = [
                        torch.tensor(0.0, device=query.device)
                    ] * 4
                    best_sim_q = torch.zeros(3, device=query.device)
                 
                print(
                    f"[Monitor] MaxOffset: {self.max_offset:.1f} | "
                    f"Entropy: {entropy.item():.3f} | "
                    f"MaxAttn: {max_attn.item():.3f} | "
                    f"Sim-STD: {sim_std.item():.4f} | "
                    f"Dot-STD: {dot_std.item():.4f} | "
                    f"BestSimMean: {best_sim_mean.item():.4f} | "
                    f"BestSimSTD: {best_sim_std.item():.4f} | "
                    f"BestSimP10/P50/P90: "
                    f"{best_sim_q[0].item():.4f}/"
                    f"{best_sim_q[1].item():.4f}/"
                    f"{best_sim_q[2].item():.4f} | "
                    f"BestSimMin/Max: {best_sim_min.item():.4f}/{best_sim_max.item():.4f} | "
                    f"Tau: {self.tau.item():.3f} | "
                    f"Offset: {off_dist.item():.3f}px"
                )

        # 15. aggregation
        v_flat = v.permute(0, 1, 2, 3, 5, 4)
        out = (attn_weights.reshape(B, Lq, self.num_heads, 1, -1) @ v_flat.reshape(B, Lq, self.num_heads, -1, self.head_dim)).squeeze(-2)
        out = out.masked_fill(no_value.unsqueeze(-1), 0.0)
        out = out.view(B, Lq, self.embed_dim)
        x = self.output_proj(out)
        pixel_no_value = no_value.all(dim=-1)
        x = x.masked_fill(pixel_no_value.unsqueeze(-1), 0.0)
        if evidence_gate is not None:
            pixel_gate = evidence_gate.mean(dim=2).unsqueeze(-1)
            x = x * pixel_gate

        if self.fusion_mode == "raw_add":
            if residual_query is None:
                raise ValueError(
                    "fusion_mode='raw_add' requires residual_query"
                )
            residual = residual_query.permute(0, 2, 3, 1).reshape(
                B,
                Lq,
                C,
            )
            res = residual + x
        else:
            gate_scale = torch.sigmoid(self.gate - 3.5)
            if (not self.training) or torch.rand(1).item() < 0.005:
                with torch.no_grad():
                    mode = "Train" if self.training else "Eval"
                    print(
                        f"[Gate-{mode}] MaxOffset: {self.max_offset:.1f} | "
                        f"GlobalGate: {gate_scale.item():.4f}"
                        + (
                            f" | EvidenceGateMean: {evidence_gate.mean().item():.4f}"
                            f" | EvidenceGateMin: {evidence_gate.min().item():.4f}"
                            f" | EvidenceGateMax: {evidence_gate.max().item():.4f}"
                            if evidence_gate is not None
                            else ""
                        )
                    )
            res = q_norm + gate_scale * x
            res = res + self.ffn(self.norm_ffn(res))
        out_feat = res.view(B, Hq, Wq, C).permute(0, 3, 1, 2)
        
        if output_attn or return_aux:
            cam_attn = attn_weights.sum(dim=-1)
            extras = {}
            if output_attn:
                extras["attn"] = cam_attn.detach()
                cam_all_invalid = invalid.all(dim=-1)
                cam_cosine = sim_logits.masked_fill(
                    invalid, float("-inf")
                ).max(dim=-1).values
                cam_cosine = cam_cosine.masked_fill(
                    cam_all_invalid, float("nan")
                )
                extras["cosine"] = cam_cosine.detach()
            if return_aux and evidence_logits is not None:
                extras["evidence_logits"] = best_evidence.permute(
                    0, 2, 1
                ).reshape(B, self.num_heads, Hq, Wq)
                extras["evidence_gate"] = evidence_gate.permute(
                    0, 2, 1
                ).reshape(B, self.num_heads, Hq, Wq)
                extras["evidence_valid"] = (~all_invalid).permute(
                    0, 2, 1
                ).reshape(B, self.num_heads, Hq, Wq)
            return out_feat, extras
        return out_feat

class MultiScaleDeformableFusion(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_cameras=5,
        num_scales=4,
        num_points=4,
        dropout=0.1,
        dot_weight=0.0,
        use_evidence_gate=False,
        evidence_score_weight=0.2,
        value_mode="projected",
        fusion_mode="standard",
        null_support_threshold=0.0,
        null_support_min_cameras=2,
    ):
        super().__init__()
        self.num_scales, self.num_cameras = num_scales, num_cameras
        self.scale_keys = ["res2", "res3", "res4", "res5"]
        ranges = [2.5, 3.0, 2.0, 1.0] 
        self.blocks = nn.ModuleList([
            PyTorchDeformableCrossAttentionBlock(
                embed_dim,
                num_heads,
                num_cameras,
                num_points,
                dropout=dropout,
                max_offset=ranges[i],
                dot_weight=dot_weight,
                use_evidence_gate=use_evidence_gate,
                evidence_score_weight=evidence_score_weight,
                value_mode=value_mode,
                fusion_mode=fusion_mode,
                null_support_threshold=null_support_threshold,
                null_support_min_cameras=null_support_min_cameras,
            )
            for i in range(num_scales)
        ])
        self.scale_embed = nn.Embedding(num_scales, embed_dim)

    def forward(
        self,
        tdom_features,
        cam_features,
        cam_exist=None,
        output_attn=False,
        return_aux=False,
    ):
        fused_features, attns, auxiliary = [], {}, {}
        for i, scale_key in enumerate(self.scale_keys):
            raw_query = tdom_features[i]
            query = raw_query + self.scale_embed.weight[i].view(1, -1, 1, 1)
            cam_list = []
            for cam_idx in range(self.num_cameras):
                cf = cam_features[cam_idx][scale_key]
                if cf.shape[-2:] != query.shape[-2:]:
                    cf = F.interpolate(cf, size=query.shape[-2:], mode='bilinear', align_corners=False)
                cam_list.append(cf)
            cam_stack = torch.stack(cam_list, dim=1) 
            if output_attn or return_aux:
                out, extras = self.blocks[i](
                    query,
                    cam_stack,
                    cam_exist,
                    output_attn=output_attn,
                    return_aux=return_aux,
                    residual_query=raw_query,
                )
                if output_attn:
                    attns[self.scale_keys[i]] = extras
                if return_aux:
                    auxiliary[self.scale_keys[i]] = extras
            else:
                out = self.blocks[i](
                    query,
                    cam_stack,
                    cam_exist,
                    residual_query=raw_query,
                )
            fused_features.append(out)
        if output_attn and return_aux:
            return fused_features, attns, auxiliary
        if output_attn:
            return fused_features, attns
        if return_aux:
            return fused_features, auxiliary
        return fused_features
