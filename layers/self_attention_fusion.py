import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.cross_attention_fusion import GlobalSpatialSelfAttention

class SelfAttentionTokenMixingBlock(nn.Module):
    """
    基于 Token Mixing 的自注意力融合块
    将 TDOM (第0个Token) 与多相机/多聚类像方特征 (第1~N个Token) 拼接在一起
    进行全局自注意力计算，让特征互相印证，最后只输出更新后的 TDOM Token。
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, ffn_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Pre-norm 分别保留 TDOM 和像方的分布
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        # 统一映射
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # FFN
        self.norm_ffn = nn.LayerNorm(embed_dim)
        ffn_hidden = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim),
            nn.Dropout(dropout),
        )

        self.attn_drop = nn.Dropout(dropout)
        self.drop_out = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor = None, all_masked: torch.Tensor = None, output_attn: bool = False):
        """
        Args:
            query: [B, Nq, C]  — TDOM 特征 (flatten)
            kv:    [B, Nq, Nkv, C] — 每个 Query 对应的像方候选
            kv_mask: [B, Nq, Nkv] bool mask, True=无效特征 (可选)
        """
        B, Nq, C = query.shape
        Nkv = kv.shape[2]

        # 防止无效特征的信息泄漏
        if kv_mask is not None:
            kv = kv.masked_fill(kv_mask.unsqueeze(-1), 0.0)

        # 1. Norm 分别处理，保留各自的特征尺度
        query_normed = self.norm_q(query).unsqueeze(2)  # [B, Nq, 1, C]
        kv_normed = self.norm_kv(kv)                    # [B, Nq, Nkv, C]

        # 2. 拼接为同一序列 (大家平起平坐)
        L = 1 + Nkv
        tokens_normed = torch.cat([query_normed, kv_normed], dim=2)  # [B, Nq, L, C]
        tokens_normed = tokens_normed.reshape(B * Nq, L, C)          # [B*Nq, L, C]

        # 3. 统一生成 Q, K, V
        q = self.q_proj(tokens_normed)  # [B*Nq, L, C]
        k = self.k_proj(tokens_normed)  # [B*Nq, L, C]
        v = self.v_proj(tokens_normed)  # [B*Nq, L, C]

        # Multi-head reshape
        q = q.view(B * Nq, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B*Nq, h, L, d]
        k = k.view(B * Nq, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B*Nq, h, L, d]
        v = v.view(B * Nq, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B*Nq, h, L, d]

        # 4. 全相交 Attention (互相查阅、对齐共识)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*Nq, h, L, L]

        # 5. Masking
        if kv_mask is not None:
            mask_kv = kv_mask.view(B * Nq, Nkv)
            # 第0个 Token (TDOM) 永远是基础存在的
            mask_tdom = torch.zeros((B * Nq, 1), dtype=torch.bool, device=mask_kv.device)
            full_mask = torch.cat([mask_tdom, mask_kv], dim=1)
            # 遮挡无效的 Key
            attn = attn.masked_fill(full_mask.view(B * Nq, 1, 1, L), -1e9)

        attn = F.softmax(attn, dim=-1)
        
        # 提取第0个Token对其他像方Token的注意力（用于分析可视化）
        attn_out = attn[:, :, 0, 1:].detach().reshape(B, Nq, self.num_heads, Nkv) if output_attn else None
        
        attn = self.attn_drop(attn)

        # 6. 聚合
        x = (attn @ v).transpose(1, 2).reshape(B * Nq, L, self.embed_dim)  # [B*Nq, L, C]
        
        # 7. 提取第0个Token (它已经吸收了其他经过筛选的像方特征)
        out_query = x[:, 0, :]  # [B*Nq, C]
        out_query = out_query.view(B, Nq, self.embed_dim)

        out_query = self.out_proj(out_query)
        out_query = self.drop_out(out_query)

        # 极端情况：如果像方全被 mask，增量置零
        if all_masked is not None:
            out_query = out_query.masked_fill(all_masked.view(B, 1, 1), 0.0)

        # 8. 标准残差融合 (移除人工门控，让 Attention 完全自治)
        query = query + out_query
        
        # 9. FFN
        query = query + self.ffn(self.norm_ffn(query))

        if output_attn:
            return query, attn_out
        return query


class MultiScaleSelfAttentionFusion(nn.Module):
    """
    多尺度自注意力融合模块 (Self-Attention Token Mixing 版)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_cameras: int = 5,
        num_scales: int = 4,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        use_global_attn: bool = False,
        global_attn_scales: list = [2, 3],
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.num_scales = num_scales
        self.scale_keys = ["res2", "res3", "res4", "res5"]

        # 使用全新的 Self-Attention Block
        self.fusion_blocks = nn.ModuleList([
            SelfAttentionTokenMixingBlock(embed_dim, num_heads, ffn_ratio, dropout)
            for _ in range(num_scales)
        ])

        self.region_size = 3
        
        self.local_pos_embed = nn.Parameter(torch.zeros(self.region_size**2, embed_dim))
        nn.init.trunc_normal_(self.local_pos_embed, std=.02)
        
        self.scale_embed = nn.Embedding(num_scales, embed_dim)

        self.use_global_attn = use_global_attn
        self.global_attn_scales = global_attn_scales
        if self.use_global_attn:
            self.global_attn_blocks = nn.ModuleDict()
            for scale_idx in self.global_attn_scales:
                self.global_attn_blocks[str(scale_idx)] = GlobalSpatialSelfAttention(
                    embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
                )

    def forward(self, tdom_features: list, cam_features: list, cam_exist=None, output_attn: bool = False):
        fused_features = []
        all_attns = {} if output_attn else None

        for scale_idx, scale_key in enumerate(self.scale_keys):
            tdom_feat = tdom_features[scale_idx]
            B, C, H, W = tdom_feat.shape

            query = tdom_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            query = query + self.scale_embed.weight[scale_idx].unsqueeze(0).unsqueeze(0)

            kv_list = []
            mask_list = []
            kv_hw = H * W

            for cam_idx in range(self.num_cameras):
                cam_feat = cam_features[cam_idx][scale_key]

                if cam_feat.shape[-2:] != (H, W):
                    cam_feat = F.interpolate(cam_feat, size=(H, W), mode='bilinear', align_corners=False)

                padding = self.region_size // 2
                cam_unfolted = F.unfold(cam_feat, kernel_size=self.region_size, padding=padding)
                cam_flat = cam_unfolted.view(B, C, self.region_size**2, H*W).permute(0, 3, 2, 1)
                
                feat_valid_mask = (cam_flat.abs().sum(dim=-1) > 1e-4) # [B, H*W, 9]

                cam_flat = cam_flat + self.scale_embed.weight[scale_idx].view(1, 1, 1, C)
                cam_flat = cam_flat + self.local_pos_embed.view(1, 1, self.region_size**2, C)

                kv_list.append(cam_flat)

                if cam_exist is not None:
                    cam_missing = ~(cam_exist[:, cam_idx].bool())
                    cam_mask = cam_missing.view(B, 1, 1) | (~feat_valid_mask)
                else:
                    cam_mask = ~feat_valid_mask
                
                mask_list.append(cam_mask)

            kv = torch.stack(kv_list, dim=2)               # [B, H*W, N_cam, 9, C]
            kv = kv.reshape(B, kv_hw, -1, C)               # [B, H*W, N_cam*9, C]
            
            kv_mask = torch.stack(mask_list, dim=2)        # [B, H*W, N_cam, 9]
            kv_mask = kv_mask.reshape(B, kv_hw, -1)        # [B, H*W, N_cam*9]

            all_masked = kv_mask.reshape(B, -1).all(dim=1) # [B]
            has_all_masked = all_masked.any()
            if has_all_masked:
                kv_mask = kv_mask.clone()
                kv_mask[all_masked, :, 0] = False

            # 调用新的 SelfAttentionTokenMixingBlock
            if output_attn:
                out, attn = self.fusion_blocks[scale_idx](query, kv, kv_mask, all_masked, output_attn=True)
                all_attns[scale_key] = attn
            else:
                out = self.fusion_blocks[scale_idx](query, kv, kv_mask, all_masked)

            out = out.permute(0, 2, 1).reshape(B, C, H, W)

            if self.use_global_attn and (scale_idx in self.global_attn_scales):
                out = self.global_attn_blocks[str(scale_idx)](out)

            fused_features.append(out)

        if output_attn:
            return fused_features, all_attns
        return fused_features
