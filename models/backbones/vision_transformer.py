import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

from models.backbones.swin_transformer import *

# 辅助函数，将一个元组或列表的元素相乘
def _ntuple(n):
    def parse(x):
        if isinstance(x, (int, float)):
            return tuple([x] * n)
        return x

    return parse

to_2tuple = _ntuple(2)

# --- ViT 核心组件 ---

class PatchEmbed(nn.Module):
    """
    将图像切分成图像块，并进行线性投影和位置嵌入。
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=2, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x) # (B, E, H', W')
        x = x.flatten(2).transpose(1, 2) # (B, N, E)
        return x

class Mlp(nn.Module):
    """
    多层感知机 (MLP)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    自注意力机制
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer 编码器，由多层 TransformerEncoderLayer 组成
    """
    def __init__(self, dim, num_layers, num_heads, mlp_ratio, qkv_bias, drop, attn_drop,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VisionTransformer(nn.Module):
    """
    标准的 Vision Transformer 模型
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=2,  embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.norm_layer = norm_layer
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.encoder = TransformerEncoder(
            dim=embed_dim,
            num_layers=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            act_layer=act_layer
        )

        self.norm = norm_layer(embed_dim)
        self.proj1 = nn.Conv2d(768, 96, 1)
        self.proj2 = nn.Conv2d(768, 192, 1)
        self.proj3 = nn.Conv2d(768, 384, 1)
        self.proj4 = nn.Conv2d(768, 768, 1)

    def forward(self, x):
        B = x.shape[0]
        # x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.encoder(x)
        x = self.norm(x)

        x = x[:, 1:, :]  # (B, num_patches, embed_dim)
        num_patches = x.shape[1]
        H_patch = W_patch = int(num_patches ** 0.5)
        feat_map = x.transpose(1, 2).reshape(B, -1, H_patch, W_patch)  # (B, embed_dim, H_patch, W_patch)

        c1 = F.interpolate(self.proj1(feat_map), scale_factor=4, mode='bilinear', align_corners=False)   # [B,96,56,56]
        c2 = F.interpolate(self.proj2(feat_map), scale_factor=2, mode='bilinear', align_corners=False)   # [B,192,28,28]
        c3 = self.proj3(feat_map)                                                                         # [B,384,14,14]
        c4 = F.interpolate(self.proj4(feat_map), scale_factor=0.5, mode='bilinear', align_corners=False) # [B,768,7,7]

        feats = [c1, c2, c3, c4]
        return feats



class PatchEmbedMultiView(nn.Module):
    """
    将图像切分成 patch 并映射到 embedding 空间
    """
    def __init__(self, in_chans=3, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, N, C, H, W)
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)  # 合并 batch 和 view
        x = self.proj(x)  # (B*N, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B*N, num_patches, D)
        x = x.view(B, N, -1, x.size(-1))  # (B, N, num_patches, D)
        return x  # (B, N, P, D)


class DOMGuidedFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, num_views=21):
        super().__init__()
        self.view_embed = nn.Embedding(num_views, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.cross_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x, view_index, mask=None):
        """
        x: (B, N, P, D)  -> patch embedding
        view_index: (B, N) -> 每个视角索引 (0=DOM)
        mask: (B, N) -> True=有效, False=padding
        """
        B, N, P, D = x.shape
        outputs = []

        for p in range(P):  # 每个 patch 独立融合
            patch_seq = x[:, :, p, :]  # (B, N, D)
            patch_seq = patch_seq + self.view_embed(view_index)  # 加视角编码

            # 找到 DOM 特征 (view_index=0)
            dom_feat = patch_seq[torch.arange(B), (view_index == 0).nonzero(as_tuple=True)[1]]  # (B, D)

            # 让其它视角特征 cross-attend DOM
            seq = torch.cat([dom_feat.unsqueeze(1), patch_seq], dim=1)  # (B, 1+N, D)
            seq = seq.transpose(0, 1)  # (1+N, B, D)

            fused_seq = self.cross_attn(seq, src_key_padding_mask=None)  # (1+N, B, D)
            fused_seq = fused_seq.transpose(0, 1)  # (B, 1+N, D)

            # 取出 DOM 位置的融合结果 (第 0 个 token)
            fused = fused_seq[:, 0, :]  # (B, D)
            outputs.append(fused)

        outputs = torch.stack(outputs, dim=1)  # (B, P, D)
        return outputs


class MultiViewPatchViT(nn.Module):
    """
    完整流程：Patch Embedding -> 局部多视角 Transformer -> 融合特征
    """
    def __init__(self, in_chans=3, img_size=224, patch_size=16, embed_dim=256, num_views=21, num_heads=4, num_layers=2):
        super().__init__()
        self.patch_embed = PatchEmbedMultiView(in_chans, patch_size, embed_dim)
        self.local_fuser = DOMGuidedFusion(embed_dim, num_heads, num_layers, num_views)

        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.to_image = nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

    def forward(self, image, view_index, mask=None):
        """
        image: (B, N, C, H, W)
        view_index: (B, N)
        mask: (B, N)
        """
        B, N, C, H, W = image.shape
        x = self.patch_embed(image)  # (B, N, P, D)
        fused_patches = self.local_fuser(x, view_index, mask)  # (B, P, D)
        h = w = self.img_size // self.patch_size
        D = fused_patches.shape[-1]
        fused_map = fused_patches.transpose(1, 2).view(B, D, h, w)  # (B, D, h, w)
        # 将 patch-level 融合特征还原回图像
        # patches = self.to_image(fused_patches)  # (B, P, C*ps*ps)
        # patches = patches.view(B, -1, self.in_chans, self.patch_size, self.patch_size)  # (B, P, C, ps, ps)

        # h = w = self.img_size // self.patch_size
        # patches = patches.view(B, h, w, self.in_chans, self.patch_size, self.patch_size)
        # patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        # fused_image = patches.view(B, self.in_chans, self.img_size, self.img_size)

        return fused_map



class MultiViewViTBackbone(nn.Module):
    def __init__(self, multi_vit: dict, vit: dict):
        super().__init__()
        # multi_vit dict 中包含 MultiImageWholeViT 的所有参数
        self.multi_vit = MultiViewPatchViT(**multi_vit)
        # vit dict 中包含 VisionTransformer 的所有参数
        self.vit = SwinTransformer(**vit)

    def forward(self, x):
        image, view_index, mask = x['image'], x['view_index'], x['mask']
        fused_image= self.multi_vit(image, view_index, mask)
        feat_map = self.vit(fused_image)
        return feat_map
   
   
   
# class PatchEmbed_mulitview(nn.Module):
#     """
#     将图像切分成图像块，并进行线性投影和位置嵌入。
#     """
#     def __init__(self, in_chans, img_size, embed_dim):
#         super().__init__()
#         # img_size = to_2tuple(img_size)
#         # patch_size = to_2tuple(patch_size)
#         # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         # self.patch_size = patch_size
#         # self.num_patches = num_patches

#         self.proj = nn.Linear(in_chans * img_size * img_size, embed_dim)

#     def forward(self, x):
#         # x: (B, C, H, W)
#         B, N, C, H, W = x.shape
#         x = x.view(B, N, -1)
#         x = self.proj(x) # (B, E, H', W')
#         return x
    
    
# class MultiImageWholeViT(nn.Module):
#      def __init__(self, in_chans=2, img_size=224, embed_dim=25088, num_views=21, num_heads=8, num_layers=6):
#             super().__init__()
#             self.embed = PatchEmbed_mulitview(in_chans, img_size, embed_dim)
#             self.view_embed = nn.Embedding(num_views, embed_dim)

#             encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
#             self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#             # attention pooling
#             self.attn_pool = nn.Linear(embed_dim, 1)  # 学习每个视角的权重
#             self.to_image = nn.Linear(embed_dim, in_chans * img_size * img_size)

#      def forward(self, image, view_index, mask):
#             """
#             image: (B, N, C, H, W)
#             view_index: (B, N)
#             mask: (B, N) bool, True=有效, False=padding
#             """
#             B, N, C, H, W = image.shape

#             # ---- 1. Whole image embedding ----
#             x = self.embed(image)  # (B, N, D)

#             # ---- 2. Add view embedding ----
#             x = x + self.view_embed(view_index)  # (B, N, D)

#             # ---- 3. Transformer encoder ----
#             x = self.transformer(x.transpose(0, 1), src_key_padding_mask=~mask)  # (N, B, D)
#             x = x.transpose(0, 1)  # (B, N, D)

#             # ---- 4. Attention pooling over views ----
#             attn_scores = self.attn_pool(x).squeeze(-1)  # (B, N)
#             attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
#             attn_weights = F.softmax(attn_scores, dim=1)  # (B, N)

#             fused = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, D)

#             # ---- 5. Project fused feature back to image size ----
#             fused_image = self.to_image(fused)  # (B, C*H*W)
#             fused_image = fused_image.view(B, C, H, W)  # (B, C, H, W)

#             return fused_image, x, attn_weights
        
# class LocalMultiViewTransformer(nn.Module):
#     """
#     局部多视角特征融合 (每个 patch 的多视角序列 -> Transformer 融合)
#     """
#     def __init__(self, embed_dim=256, num_heads=4, num_layers=2, num_views=21):
#         super().__init__()
#         self.view_embed = nn.Embedding(num_views, embed_dim)

#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.attn_pool = nn.Linear(embed_dim, 1)  # attention pooling

#     def forward(self, x, view_index, mask=None):
#         """
#         x: (B, N, P, D)  -> patch embedding
#         view_index: (B, N) -> 每个视角的索引
#         mask: (B, N) bool -> True=有效, False=padding
#         """
#         B, N, P, D = x.shape
#         outputs = []

#         for p in range(P):  # 遍历每个 patch 位置
#             patch_seq = x[:, :, p, :]  # (B, N, D)
#             patch_seq = patch_seq + self.view_embed(view_index)  # 加视角编码

#             patch_seq = patch_seq.transpose(0, 1)  # (N, B, D)
#             patch_fused = self.transformer(patch_seq, src_key_padding_mask=(~mask if mask is not None else None))  # (N, B, D)
#             patch_fused = patch_fused.transpose(0, 1)  # (B, N, D)

#             attn_scores = self.attn_pool(patch_fused).squeeze(-1)  # (B, N)
#             if mask is not None:
#                 attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
#             attn_weights = F.softmax(attn_scores, dim=1)  # (B, N)

#             fused = torch.sum(patch_fused * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
#             outputs.append(fused)

#         outputs = torch.stack(outputs, dim=1)  # (B, P, D)
#         return outputs  # 每个 patch 的融合特征
 
# class MultiViewViTBackbone(nn.Module):
#     def __init__(self, multi_vit: dict):
#         super().__init__()
#         # multi_vit dict 中包含 MultiImageWholeViT 的所有参数
#         self.multi_vit = MultiViewPatchViT(**multi_vit)
#         # vit dict 中包含 VisionTransformer 的所有参数
#         # self.vit = SwinTransformer(**vit)
#         embed_dim = multi_vit.get('embed_dim', 768)
#         self.conv1 = nn.Conv2d(embed_dim, 96, kernel_size=1)
#         self.conv2 = nn.Conv2d(embed_dim, 192, kernel_size=1)
#         self.conv3 = nn.Conv2d(embed_dim, 384, kernel_size=1)
#         self.conv4 = nn.Conv2d(embed_dim, 768, kernel_size=1)
#         self.pool = nn.AdaptiveAvgPool2d

#     def forward(self, x):
#         image, view_index, mask = x['image'], x['view_index'], x['mask']
#         fused_image, _ = self.multi_vit(image, view_index, mask)
#         c1 = self.conv1(fused_image)  # 保持原尺寸
#         c2 = self.conv2(self.pool((c1.size(2)//2, c1.size(3)//2))(fused_image))
#         c3 = self.conv3(self.pool((c1.size(2)//4, c1.size(3)//4))(fused_image))
#         c4 = self.conv4(self.pool((c1.size(2)//8, c1.size(3)//8))(fused_image))
#         # feat_map = self.vit(fused_image)
#         return [c1, c2, c3, c4]
