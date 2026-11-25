import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Tuple,List
from functools import partial
from utils.pth_pca import vis_pca_feature
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from models.modules import upsample_x_to_fit_y
from hook.basehook import HookClass
from models.backbones import *
from models.heads import *
from utils.basic_func import load_pretrain_with_checkpoint
from layers.spatial_fusion import SpatialFusion2DV2, FusionModule
from layers.conv_module import ConvModule
from peft import LoraConfig, get_peft_model
__all__ = ['MVMapper','EncoderDecoder','EncoderDecoderCrossAttn','SegmentationModel']


def get_mode_cal(mode='wise'):
    func_mode={'cat':partial(torch.cat,dim=1),'wise':sum}
    return func_mode[mode]


def compute_dsm_features(dsm, kernel_size=65, rel_clip=40.0, grad_clip=10.0):
    dsm = torch.nan_to_num(dsm, nan=0.0, posinf=0.0, neginf=0.0)
    pad = kernel_size // 2

    local_min = -F.max_pool2d(-dsm, kernel_size=kernel_size, stride=1, padding=pad)
    relative_height = (dsm - local_min).clamp(min=0.0, max=rel_clip) / rel_clip

    dx = F.pad(dsm[:, :, :, 1:] - dsm[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(dsm[:, :, 1:, :] - dsm[:, :, :-1, :], (0, 0, 0, 1))
    dsm_gradient = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-6)
    dsm_gradient = dsm_gradient.clamp(min=0.0, max=grad_clip) / grad_clip

    return relative_height, dsm_gradient


def build_spatial_input(neck, image, dsm):
    spatial_in_channels = getattr(neck, "spatial_in_channels", 4)
    if spatial_in_channels == 4:
        return torch.cat([image, dsm], dim=1)
    if spatial_in_channels == 6:
        relative_height, dsm_gradient = compute_dsm_features(dsm)
        return torch.cat([image, dsm, relative_height, dsm_gradient], dim=1)
    raise ValueError(
        f"Unsupported spatial_in_channels={spatial_in_channels}. "
        "Use 4 for RGB+DSM or 6 for RGB+DSM+relative_height+dsm_gradient."
    )
# upernet
# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder, out_class,neck=None, **kwargs):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         lora_config = LoraConfig(
#             r=16,
#             lora_alpha=32,
#             target_modules=["qkv"], 
#             lora_dropout=0.1,
#         )
#         # 把 LoRA 注入到 backbone 中
#         # 注意：这里我们只包 backbone，保留外面的 wrapper 结构
#         self.encoder.backbone = get_peft_model(self.encoder.backbone, lora_config)
        
#         # 打印一下看看有多少参数是活的
#         self.encoder.backbone.print_trainable_parameters()
#         self.neck = neck
#         self.decoder = decoder
#         self.out_class = out_class
#         self.conv = nn.Conv2d(self.decoder.out_channels, self.out_class, 1)
#     def forward(self, x, proj_feature = None):
#         image,dsm=x
#         _, vit_features = self.encoder(image)
#         spatial_input = torch.cat([image, dsm], dim=1) 
#         if proj_feature is not None:
#             if isinstance(proj_feature, (tuple,list)):
#                 features = [x + y for x, y in zip(features, proj_feature)]
#             else:
#                 features += proj_feature
#         if self.neck is not None:
#             features = self.neck((spatial_input, vit_features))
#         final_feature = self.decoder(features)
#         out = upsample_x_to_fit_y(x=final_feature, y=image)
#         out = self.conv(out)
#         return out


# 特征融合

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, out_class, neck=None, **kwargs):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv"],
        )
        self.encoder.backbone = get_peft_model(self.encoder.backbone, lora_config)
        self.encoder.backbone.print_trainable_parameters()

        self.neck = neck
        self.decoder = decoder
        self.out_class = out_class
        self.conv = nn.Conv2d(self.decoder.out_channels, self.out_class, 1)
        # Keep the historical direct-add behavior by default.  New memory
        # experiments can opt into a small learnable residual via config.
        self.fusion_type = str(kwargs.get("fusion_type", "legacy_add")).lower()
        alpha_init = kwargs.get("fusion_alpha", kwargs.get("alpha_init", 1.0))
        if isinstance(alpha_init, (list, tuple)):
            alpha_values = [float(v) for v in alpha_init]
        else:
            alpha_values = [float(alpha_init)] * 4
        learnable_alpha = bool(kwargs.get("learnable_fusion_alpha", self.fusion_type not in {"legacy_add", "direct_add"}))
        alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32)
        if learnable_alpha:
            self.fusion_alpha = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer("fusion_alpha", alpha_tensor)

    def forward(self, x, proj_feature=None):
        image, dsm = x
        _, vit_features = self.encoder(image)  # dict: [B,C,H,W]

        spatial_input = build_spatial_input(self.neck, image, dsm)

        if self.neck is not None:
            neck_out = self.neck((spatial_input, vit_features))
        else:
            neck_out = vit_features

        # 像方 + 物方特征融合（同 batch、同尺度）
        if proj_feature is not None:
            fused_features = {}
            scale_keys = ["res2", "res3", "res4", "res5"]
            for scale_id, k in enumerate(neck_out.keys()):
                if self.fusion_type in {"legacy_add", "direct_add"}:
                    alpha = self.fusion_alpha[min(scale_id, len(self.fusion_alpha) - 1)]
                elif self.fusion_type in {"residual_add", "scaled_add", "image_memory_mean"}:
                    alpha = self.fusion_alpha[min(scale_id, len(self.fusion_alpha) - 1)]
                else:
                    raise ValueError(
                        f"Unsupported fusion_type={self.fusion_type!r}; "
                        "use legacy_add or residual_add."
                    )
                fused_features[k] = neck_out[k] + alpha.to(dtype=neck_out[k].dtype, device=neck_out[k].device) * proj_feature[k]
        else:
            fused_features = neck_out

        decoder_inputs = [fused_features[k] for k in ["res2","res3","res4","res5"] if k in fused_features]

        final_feature = self.decoder(decoder_inputs)
        out = upsample_x_to_fit_y(final_feature, image)
        out = self.conv(out)

        return out


class EncoderDecoderCrossAttn(nn.Module):
    """
    与 EncoderDecoder 相同的编码-解码结构，但在像方特征与物方特征的
    融合环节使用 Multi-Scale Cross-Attention 替代简单的逐元素相加。

    forward 输入:
        x: (image, dsm)
        cam_features: list[N_cam], 每个元素 dict{res2/3/4/5: [B,C,H,W]}
        cam_exist: Tensor[B, N_cam] bool  — 标记每个 batch 中各相机是否存在
    """

    def __init__(self, encoder, decoder, out_class, neck=None,
                 cross_attn_cfg=None, **kwargs):
        super().__init__()
        self.encoder = encoder

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv"],
        )
        self.encoder.backbone = get_peft_model(self.encoder.backbone, lora_config)
        self.encoder.backbone.print_trainable_parameters()

        self.neck = neck
        self.decoder = decoder
        self.out_class = out_class
        self.conv = nn.Conv2d(self.decoder.out_channels, self.out_class, 1)
        self.freeze_feature_extractor = bool(kwargs.get("freeze_feature_extractor", False))
        self.freeze_tdom_branch = bool(kwargs.get("freeze_tdom_branch", False))

        feature_pretrained_ckpt = kwargs.get("feature_pretrained_ckpt", None)
        if self.freeze_feature_extractor and not feature_pretrained_ckpt:
            raise ValueError("freeze_feature_extractor=True requires model.feature_pretrained_ckpt.")
        if feature_pretrained_ckpt:
            self.load_feature_extractor_from_checkpoint(feature_pretrained_ckpt)
        tdom_pretrained_ckpt = kwargs.get("tdom_pretrained_ckpt", None)
        if tdom_pretrained_ckpt:
            self.load_tdom_branch_from_checkpoint(tdom_pretrained_ckpt)
        if self.freeze_feature_extractor:
            self.freeze_encoder_neck()

        # Point-to-Region Cross-Attention Fusion Module
        if cross_attn_cfg is None:
            cross_attn_cfg = {}
        from layers.cross_attention_fusion import MultiScaleCrossAttentionFusion
        from layers.self_attention_fusion import MultiScaleSelfAttentionFusion
        from layers.deformable_fusion import MultiScaleDeformableFusion
        from layers.candidate_gate_fusion import MultiScaleCandidateGateFusion
        from layers.image_memory_fusion import MaskedMeanMemoryFusion
        from layers.sail_image_memory import SAILImageMemoryFusion

        fusion_type = cross_attn_cfg.get('fusion_type', '').lower()
        self._uses_sail_image_memory = fusion_type in {
            'sail_image_memory', 'sail_memory', 'sail_lite'
        }
        offline_ablation = fusion_type in {'offline_allview_max_concat', 'offline_cluster21_tdom_attention'}
        self._uses_tdom_joint_view_attention = fusion_type == 'tdom_joint_view_attention' or offline_ablation
        if offline_ablation:
            if cross_attn_cfg.get('final_fusion', 'concat_project') != 'concat_project':
                raise ValueError('Offline ablations require concat_project fusion')
            if cross_attn_cfg.get('mask_zero_tail_features', False):
                raise ValueError('Offline ablations retain real zero-valued features')
            from layers.offline_allview_ablation import OfflineMaxConcatFusion, OfflineCluster21AttentionFusion
            common = dict(embed_dim=cross_attn_cfg.get('embed_dim', 256),
                          num_scales=cross_attn_cfg.get('num_scales', 4),
                          alpha_init=cross_attn_cfg.get('alpha_init', 0.05))
            if fusion_type == 'offline_allview_max_concat':
                self.cross_attn_fusion = OfflineMaxConcatFusion(**common)
            else:
                self.cross_attn_fusion = OfflineCluster21AttentionFusion(
                    **common, num_heads=cross_attn_cfg.get('num_heads', 8),
                    ffn_ratio=cross_attn_cfg.get('ffn_ratio', 2.0),
                    dropout=cross_attn_cfg.get('dropout', 0.0),
                    query_chunk_size=cross_attn_cfg.get('query_chunk_size', 512))
        elif self._uses_tdom_joint_view_attention:
            if cross_attn_cfg.get('final_fusion', 'concat_project') != 'concat_project':
                raise ValueError('tdom_joint_view_attention requires matched concat_project fusion')
            if not cross_attn_cfg.get('mask_zero_tail_features', True):
                raise ValueError('tdom_joint_view_attention requires the matched zero-feature mask')
            from layers.tdom_joint_view_attention import TDOMJointViewAttentionFusion
            self.cross_attn_fusion = TDOMJointViewAttentionFusion(
                embed_dim=cross_attn_cfg.get('embed_dim', 256),
                num_heads=cross_attn_cfg.get('num_heads', 8),
                num_scales=cross_attn_cfg.get('num_scales', 4),
                ffn_ratio=cross_attn_cfg.get('ffn_ratio', 2.0),
                dropout=cross_attn_cfg.get('dropout', 0.0),
                alpha_init=cross_attn_cfg.get('alpha_init', 0.05),
                eps=cross_attn_cfg.get('memory_eps', 1e-6),
                tail_local_context_enabled=cross_attn_cfg.get('tail_local_context_enabled', True),
                tail_local_context_scales=cross_attn_cfg.get('tail_local_context_scales', ['res2', 'res3']),
                tail_local_context_kernel_size=cross_attn_cfg.get('tail_local_context_kernel_size', 5),
                tail_local_context_bottleneck_ratio=cross_attn_cfg.get('tail_local_context_bottleneck_ratio', 4),
                tail_local_context_dropout=cross_attn_cfg.get('tail_local_context_dropout', 0.0),
                tail_local_context_alpha_init=cross_attn_cfg.get('tail_local_context_alpha_init', 1.0),
                tail_local_context_chunk_size=cross_attn_cfg.get('tail_local_context_chunk_size', 4),
                query_chunk_size=cross_attn_cfg.get('query_chunk_size', 2048),
            )
        elif self._uses_sail_image_memory:
            self.cross_attn_fusion = SAILImageMemoryFusion(
                embed_dim=cross_attn_cfg.get('embed_dim', 256),
                num_heads=cross_attn_cfg.get('num_heads', 8),
                num_scales=cross_attn_cfg.get('num_scales', 4),
                anchor_depth=cross_attn_cfg.get('anchor_depth', 1),
                dropout=cross_attn_cfg.get('dropout', 0.0),
                ffn_ratio=cross_attn_cfg.get('ffn_ratio', 2.0),
                alpha_init=cross_attn_cfg.get('alpha_init', 0.05),
                tail_beta_init=cross_attn_cfg.get('tail_beta_init', 0.0),
                attention_direction=cross_attn_cfg.get(
                    'attention_direction', 'tail_to_anchor'
                ),
                use_null_tail=cross_attn_cfg.get('use_null_tail', False),
                null_tail_logit_init=cross_attn_cfg.get(
                    'null_tail_logit_init', -1.0
                ),
                tail_aggregation=cross_attn_cfg.get('tail_aggregation', 'attention'),
                tail_value_mode=cross_attn_cfg.get('tail_value_mode', 'projected'),
                final_fusion=cross_attn_cfg.get('final_fusion', 'add'),
                tail_topk=cross_attn_cfg.get('tail_topk', 8),
                tail_topk_min=cross_attn_cfg.get('tail_topk_min', 1),
                tail_topk_eval=cross_attn_cfg.get('tail_topk_eval', None),
                tail_topk_random_train=cross_attn_cfg.get(
                    'tail_topk_random_train', False
                ),
                tail_global_score_scale=cross_attn_cfg.get(
                    'tail_global_score_scale', 'res3'
                ),
                tail_score_temperature=cross_attn_cfg.get(
                    'tail_score_temperature', 1.0
                ),
                anchor_spatial_window_size=cross_attn_cfg.get(
                    'anchor_spatial_window_size', 1
                ),
                anchor_spatial_scales=cross_attn_cfg.get(
                    'anchor_spatial_scales', None
                ),
                anchor_spatial_depth=cross_attn_cfg.get('anchor_spatial_depth', 1),
                anchor_spatial_residual_init=cross_attn_cfg.get(
                    'anchor_spatial_residual_init', 0.1
                ),
                swin_r_enabled=cross_attn_cfg.get('swin_r_enabled', False),
                swin_r_scales=cross_attn_cfg.get('swin_r_scales', None),
                swin_r_window_size=cross_attn_cfg.get('swin_r_window_size', 8),
                swin_r_num_heads=cross_attn_cfg.get('swin_r_num_heads', 8),
                swin_r_depth=cross_attn_cfg.get('swin_r_depth', 2),
                swin_r_mlp_ratio=cross_attn_cfg.get('swin_r_mlp_ratio', 4.0),
                swin_r_dropout=cross_attn_cfg.get('swin_r_dropout', 0.0),
                swin_r_alpha_init=cross_attn_cfg.get('swin_r_alpha_init', 0.01),
                multi_latent_swin_r_enabled=cross_attn_cfg.get(
                    'multi_latent_swin_r_enabled', False
                ),
                multi_latent_num=cross_attn_cfg.get('multi_latent_num', 3),
                multi_latent_spatial_window_size=cross_attn_cfg.get(
                    'multi_latent_spatial_window_size', 8
                ),
                multi_latent_spatial_depth=cross_attn_cfg.get(
                    'multi_latent_spatial_depth', 2
                ),
                multi_latent_depth=cross_attn_cfg.get('multi_latent_depth', 1),
                multi_latent_beta_init=cross_attn_cfg.get(
                    'multi_latent_beta_init', 0.01
                ),
                multi_latent_alpha_init=cross_attn_cfg.get(
                    'multi_latent_alpha_init', 0.01
                ),
                single_r_swin_set_enabled=cross_attn_cfg.get(
                    'single_r_swin_set_enabled', False
                ),
                single_r_swin_set_scales=cross_attn_cfg.get(
                    'single_r_swin_set_scales', None
                ),
                single_r_spatial_window_size=cross_attn_cfg.get(
                    'single_r_spatial_window_size', 8
                ),
                single_r_spatial_depth=cross_attn_cfg.get(
                    'single_r_spatial_depth', 2
                ),
                single_r_view_depth=cross_attn_cfg.get(
                    'single_r_view_depth', 1
                ),
                single_r_beta_init=cross_attn_cfg.get(
                    'single_r_beta_init', 0.01
                ),
                single_r_spatial_alpha_init=cross_attn_cfg.get(
                    'single_r_spatial_alpha_init', 0.01
                ),
                tail_local_context_enabled=cross_attn_cfg.get(
                    'tail_local_context_enabled', False
                ),
                tail_local_context_scales=cross_attn_cfg.get(
                    'tail_local_context_scales', None
                ),
                tail_local_context_kernel_size=cross_attn_cfg.get(
                    'tail_local_context_kernel_size', 5
                ),
                tail_local_context_bottleneck_ratio=cross_attn_cfg.get(
                    'tail_local_context_bottleneck_ratio', 4
                ),
                tail_local_context_dropout=cross_attn_cfg.get(
                    'tail_local_context_dropout', 0.0
                ),
                tail_local_context_alpha_init=cross_attn_cfg.get(
                    'tail_local_context_alpha_init', 1.0
                ),
                tail_local_context_chunk_size=cross_attn_cfg.get(
                    'tail_local_context_chunk_size', 4
                ),
                r_conditioned_local_tail_enabled=cross_attn_cfg.get(
                    'r_conditioned_local_tail_enabled', False
                ),
                r_conditioned_local_tail_scales=cross_attn_cfg.get(
                    'r_conditioned_local_tail_scales', None
                ),
                r_conditioned_local_tail_window_sizes=cross_attn_cfg.get(
                    'r_conditioned_local_tail_window_sizes', None
                ),
                r_conditioned_local_tail_attention_dim=cross_attn_cfg.get(
                    'r_conditioned_local_tail_attention_dim', 64
                ),
                r_conditioned_local_tail_num_heads=cross_attn_cfg.get(
                    'r_conditioned_local_tail_num_heads', 4
                ),
                r_conditioned_local_tail_ffn_ratio=cross_attn_cfg.get(
                    'r_conditioned_local_tail_ffn_ratio', 2.0
                ),
                r_conditioned_local_tail_dropout=cross_attn_cfg.get(
                    'r_conditioned_local_tail_dropout', 0.0
                ),
                tail_anchor_memory_mode=cross_attn_cfg.get(
                    'tail_anchor_memory_mode', 'same_pixel'
                ),
                tail_anchor_local_window_size=cross_attn_cfg.get(
                    'tail_anchor_local_window_size', 1
                ),
                tail_anchor_local_scales=cross_attn_cfg.get(
                    'tail_anchor_local_scales', None
                ),
                tail_anchor_local_residual_init=cross_attn_cfg.get(
                    'tail_anchor_local_residual_init', 0.1
                ),
                tail_anchor_local_query_chunk_size=cross_attn_cfg.get(
                    'tail_anchor_local_query_chunk_size', 4
                ),
                tail_anchor_local_checkpoint=cross_attn_cfg.get(
                    'tail_anchor_local_checkpoint', True
                ),
                tdom_memory_token_enabled=cross_attn_cfg.get(
                    'tdom_memory_token_enabled', False
                ),
                tdom_memory_token_scales=cross_attn_cfg.get(
                    'tdom_memory_token_scales', None
                ),
                tdom_memory_token_hidden_ratio=cross_attn_cfg.get(
                    'tdom_memory_token_hidden_ratio', 1.0
                ),
                tdom_memory_token_readout=cross_attn_cfg.get(
                    'tdom_memory_token_readout', 'mlp'
                ),
                tdom_memory_gate_hidden_dim=cross_attn_cfg.get(
                    'tdom_memory_gate_hidden_dim', 32
                ),
                tail_dual_memory_context_enabled=cross_attn_cfg.get(
                    'tail_dual_memory_context_enabled', False
                ),
                tail_dual_memory_context_scales=cross_attn_cfg.get(
                    'tail_dual_memory_context_scales', None
                ),
                tail_dual_memory_context_ffn_ratio=cross_attn_cfg.get(
                    'tail_dual_memory_context_ffn_ratio', 2.0
                ),
                tail_dual_memory_context_alpha_init=cross_attn_cfg.get(
                    'tail_dual_memory_context_alpha_init', 0.01
                ),
                tdom_in_anchor_memory=cross_attn_cfg.get(
                    'tdom_in_anchor_memory', False
                ),
                tdom_in_anchor_memory_scales=cross_attn_cfg.get(
                    'tdom_in_anchor_memory_scales', None
                ),
                tdom_in_anchor_memory_alpha_init=cross_attn_cfg.get(
                    'tdom_in_anchor_memory_alpha_init', 0.01
                ),
                memory_output_mode=cross_attn_cfg.get(
                    'memory_output_mode', 'anchor_plus_tail'
                ),
                mask_zero_tail_features=cross_attn_cfg.get(
                    'mask_zero_tail_features', True
                ),
                tail_content_gamma=cross_attn_cfg.get('tail_content_gamma', 1.0),
                tail_content_gamma_scales=cross_attn_cfg.get(
                    'tail_content_gamma_scales', None
                ),
                eps=cross_attn_cfg.get('memory_eps', 1e-6),
            )
        elif fusion_type in {
            'image_memory_mean', 'masked_mean_memory', 'mean_memory'
        }:
            self.cross_attn_fusion = MaskedMeanMemoryFusion(
                embed_dim=cross_attn_cfg.get('embed_dim', 256),
                num_scales=cross_attn_cfg.get('num_scales', 4),
                alpha_init=cross_attn_cfg.get('alpha_init', 0.05),
                eps=cross_attn_cfg.get('memory_eps', 1e-6),
            )
        elif cross_attn_cfg.get('use_candidate_gate', False):
            self.cross_attn_fusion = MultiScaleCandidateGateFusion(
                embed_dim=cross_attn_cfg.get('embed_dim', 256),
                num_cameras=cross_attn_cfg.get('num_cameras', 5),
                num_scales=cross_attn_cfg.get('num_scales', 4),
                dropout=cross_attn_cfg.get('dropout', 0.1),
                hidden_ratio=cross_attn_cfg.get(
                    'candidate_hidden_ratio', 0.25
                ),
                null_support_min_cameras=cross_attn_cfg.get(
                    'null_support_min_cameras', 2
                ),
                null_support_score_margin=cross_attn_cfg.get(
                    'null_support_score_margin', 0.1
                ),
            )
        elif cross_attn_cfg.get('use_deformable', False):
            self.cross_attn_fusion = MultiScaleDeformableFusion(
                embed_dim=cross_attn_cfg.get('embed_dim', 256),
                num_heads=cross_attn_cfg.get('num_heads', 8),
                num_cameras=cross_attn_cfg.get('num_cameras', 5),
                num_scales=cross_attn_cfg.get('num_scales', 4),
                num_points=cross_attn_cfg.get('num_points', 4),
                dropout=cross_attn_cfg.get('dropout', 0.1),
                dot_weight=cross_attn_cfg.get('dot_weight', 0.0),
                use_evidence_gate=cross_attn_cfg.get(
                    'use_evidence_gate', False
                ),
                evidence_score_weight=cross_attn_cfg.get(
                    'evidence_score_weight', 0.2
                ),
                value_mode=cross_attn_cfg.get('value_mode', 'projected'),
                fusion_mode=cross_attn_cfg.get(
                    'fusion_mode', 'standard'
                ),
                null_support_threshold=cross_attn_cfg.get(
                    'null_support_threshold', 0.0
                ),
                null_support_min_cameras=cross_attn_cfg.get(
                    'null_support_min_cameras', 2
                ),
                null_support_score_margin=cross_attn_cfg.get(
                    'null_support_score_margin', 0.1
                ),
            )
        else:
            self.cross_attn_fusion = MultiScaleCrossAttentionFusion(
                embed_dim=cross_attn_cfg.get('embed_dim', 256),
                num_heads=cross_attn_cfg.get('num_heads', 8),
                num_cameras=cross_attn_cfg.get('num_cameras', 5),
                num_scales=cross_attn_cfg.get('num_scales', 4),
                dropout=cross_attn_cfg.get('dropout', 0.1),
                region_size=cross_attn_cfg.get('region_size', 3),
                dot_weight=cross_attn_cfg.get('dot_weight', 0.0),
                use_evidence_gate=cross_attn_cfg.get(
                    'use_evidence_gate', False
                ),
                evidence_score_weight=cross_attn_cfg.get(
                    'evidence_score_weight', 0.2
                ),
                value_mode=cross_attn_cfg.get('value_mode', 'projected'),
                null_support_threshold=cross_attn_cfg.get(
                    'null_support_threshold', 0.0
                ),
                null_support_min_cameras=cross_attn_cfg.get(
                    'null_support_min_cameras', 2
                ),
                null_support_score_margin=cross_attn_cfg.get(
                    'null_support_score_margin', 0.1
                ),
            )

        if self.freeze_tdom_branch:
            self.freeze_tdom_branch_modules()

        # Opt-in experiment: keep the independently trained Attn-R decoder
        # intact and train a separate post-decoder local tail correction.
        correction_cfg = kwargs.get("tail_logit_correction", {}) or {}
        self.tail_logit_correction_enabled = bool(correction_cfg.get("enabled", False))
        fusion_pretrained_ckpt = kwargs.get("fusion_pretrained_ckpt")
        if fusion_pretrained_ckpt:
            if not self._uses_sail_image_memory or self.tail_logit_correction_enabled:
                raise ValueError("fusion_pretrained_ckpt requires SAIL without logit correction")
            self.load_sail_fusion_checkpoint(fusion_pretrained_ckpt)
        if self.tail_logit_correction_enabled:
            from layers.tail_logit_correction import TailLogitCorrection
            fusion = self.cross_attn_fusion
            if (not self._uses_sail_image_memory
                    or fusion.attention_direction != "anchor_to_tail"
                    or fusion.tail_aggregation != "attention"
                    or fusion.tail_value_mode != "projected"
                    or not fusion.mask_zero_tail_features
                    or fusion.use_null_tail
                    or fusion.tdom_in_anchor_memory
                    or fusion.tail_dual_memory_context_scales
                    or fusion.r_conditioned_local_tail_scales
                    or fusion.memory_output_mode != "anchor_plus_tail"):
                raise ValueError("Local logit correction requires plain masked R-query tail attention")
            self.tail_logit_head = TailLogitCorrection(
                embed_dim=cross_attn_cfg.get("embed_dim", 256),
                num_scales=cross_attn_cfg.get("num_scales", 4),
                hidden_dim=correction_cfg.get("hidden_dim", 64),
                out_channels=out_class,
                gamma_init=correction_cfg.get("gamma_init", 0.1),
            )
            checkpoint = correction_cfg.get("attnr_pretrained_ckpt")
            if not checkpoint:
                raise ValueError("attnr_pretrained_ckpt is required for the frozen Attn-R baseline")
            self.load_attnr_for_logit_correction(checkpoint)
            for module_name in ("encoder", "neck", "decoder", "conv", "cross_attn_fusion"):
                module = getattr(self, module_name, None)
                if module is not None:
                    module.requires_grad_(False)
                    module.eval()
            for module in self._logit_tail_trainable_modules():
                module.requires_grad_(True)
            print("[TailLogit] Frozen Attn-R baseline; train tail readers/local context and logit head only.")

    def load_sail_fusion_checkpoint(self, checkpoint_path):
        """Warm start the COMPLETE additive model, not just the TDOM branch."""
        state = self._read_checkpoint_state(checkpoint_path)
        for name in ("encoder", "neck", "decoder", "conv", "cross_attn_fusion"):
            module = getattr(self, name, None)
            if module is None:
                continue
            weights = self._extract_module_state(state, name)
            if not weights:
                raise ValueError(f"Fusion checkpoint has no {name}: {checkpoint_path}")
            missing, unexpected = module.load_state_dict(weights, strict=False)
            bad_missing = [key for key in missing if not (
                name == "cross_attn_fusion"
                and self.cross_attn_fusion.final_fusion == "concat_project"
                and key.startswith("final_projections."))]
            if bad_missing or unexpected:
                raise RuntimeError(f"Incompatible fusion init {name}: missing={bad_missing}, unexpected={unexpected}")
            print(f"[FusionInit] {name}: {len(weights)} tensors; new projection keys={len(missing)}")
            if (name == "cross_attn_fusion" and missing
                    and self.cross_attn_fusion.final_fusion == "concat_project"):
                self.cross_attn_fusion.reset_final_projections_to_add()

    def _logit_tail_trainable_modules(self):
        fusion = self.cross_attn_fusion
        return [getattr(fusion, name) for name in (
            "tail_local_context_refiners", "tail_norm_q", "tail_norm_kv",
            "tail_attn", "tail_norm_ffn", "tail_ffn",
        )]

    def load_attnr_for_logit_correction(self, checkpoint_path):
        """Require a complete base; only newly enabled TailLocal may be missing."""
        state = self._read_checkpoint_state(checkpoint_path)
        for name in ("encoder", "neck", "decoder", "conv", "cross_attn_fusion"):
            module = getattr(self, name, None)
            if module is None:
                continue
            weights = self._extract_module_state(state, name)
            if not weights:
                raise ValueError(f"Attn-R checkpoint has no {name} weights: {checkpoint_path}")
            missing, unexpected = module.load_state_dict(weights, strict=False)
            allowed = "tail_local_context_refiners."
            bad_missing = [key for key in missing if not (
                name == "cross_attn_fusion" and key.startswith(allowed))]
            if bad_missing or unexpected:
                raise RuntimeError(f"Incomplete Attn-R init for {name}: missing={bad_missing}, unexpected={unexpected}")
            print(f"[AttnRInit] {name}: {len(weights)} tensors; new TailLocal keys={len(missing)}")

    def _forward_tail_logit_correction(self, tdom_features, image, anchors,
                                       anchor_exist, tails, tail_exist):
        fusion = self.cross_attn_fusion
        prepared, base_features = [], []
        # This branch, including the original decoder, stays in eval mode.
        with torch.no_grad():
            for sid, (scale, tdom) in enumerate(zip(("res2", "res3", "res4", "res5"), tdom_features)):
                views = fusion._stack_views(anchors, scale, tdom.device, tdom.dtype)
                state = fusion._anchor_memory(views, anchor_exist, sid, tdom)
                prepared.append(state)
                base_features.append(tdom + fusion.alpha[sid].to(tdom.dtype) * state[0])
            base = self.conv(upsample_x_to_fit_y(self.decoder(base_features), image))

        updates, eligible, aux = [], [], {
            "swin_r_alpha": [], "swin_r_delta_ratio": [],
            "anchor_memory": [], "anchor_support": [], "tail_support": [],
            "tail_local_context_alpha": [], "tail_local_context_delta_ratio": [],
        }
        for sid, (scale, tdom, state) in enumerate(zip(
                ("res2", "res3", "res4", "res5"), tdom_features, prepared)):
            memory, support = state[:2]
            raw = fusion._stack_views(tails, scale, tdom.device, tdom.dtype)
            # Keep every reader on the autograd graph even for absent/padded
            # tails. Output masking guarantees no update without evidence.
            active_exist = tail_exist
            if raw is None or raw.shape[1] == 0:
                raw = torch.zeros_like(tdom).unsqueeze(1)
                active_exist = torch.zeros((tdom.shape[0], 1), dtype=torch.bool, device=tdom.device)
            context, alpha, ratio = fusion._contextualize_tail_views(raw, active_exist, sid)
            update, tail_support = fusion._anchor_query_tail_update(
                context, active_exist, memory, support, sid, tdom, raw_tail_views=raw)
            updates.append(update)
            eligible.append((support > 0) & (tail_support > 0))
            aux["anchor_memory"].append(memory)
            aux["anchor_support"].append(support)
            aux["tail_support"].append(tail_support)
            aux["swin_r_alpha"].append(state[4])
            aux["swin_r_delta_ratio"].append(state[5])
            aux["tail_local_context_alpha"].append(alpha)
            aux["tail_local_context_delta_ratio"].append(ratio)
        out, stats = self.tail_logit_head([state[0] for state in prepared], updates, eligible, base)
        aux.update(stats)
        return out, aux

    @staticmethod
    def _read_checkpoint_state(checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"feature_pretrained_ckpt does not exist: {checkpoint_path}")
        try:
            ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt

    @staticmethod
    def _extract_module_state(state_dict, module_name):
        module_state = {}
        prefix = f"{module_name}."
        lightning_prefix = f"model.{module_name}."
        for key, value in state_dict.items():
            if key.startswith(lightning_prefix):
                module_state[key[len(lightning_prefix):]] = value
            elif key.startswith(prefix):
                module_state[key[len(prefix):]] = value
        return module_state

    def load_feature_extractor_from_checkpoint(self, checkpoint_path):
        state_dict = self._read_checkpoint_state(checkpoint_path)
        for module_name in ("encoder", "neck"):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            module_state = self._extract_module_state(state_dict, module_name)
            if not module_state:
                print(f"[FeatureInit] {module_name}: no matching keys in {checkpoint_path}")
                continue
            missing, unexpected = module.load_state_dict(module_state, strict=False)
            print(
                f"[FeatureInit] {module_name}: loaded {len(module_state)} tensors from {checkpoint_path} | "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing:
                print(f"[FeatureInit] {module_name} missing sample: {missing[:5]}")
            if unexpected:
                print(f"[FeatureInit] {module_name} unexpected sample: {unexpected[:5]}")

    def load_tdom_branch_from_checkpoint(self, checkpoint_path):
        state_dict = self._read_checkpoint_state(checkpoint_path)
        for module_name in ("encoder", "neck", "decoder", "conv"):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            module_state = self._extract_module_state(state_dict, module_name)
            if not module_state:
                print(f"[TDOMInit] {module_name}: no matching keys in {checkpoint_path}")
                continue
            missing, unexpected = module.load_state_dict(module_state, strict=False)
            print(
                f"[TDOMInit] {module_name}: loaded {len(module_state)} tensors from {checkpoint_path} | "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing:
                print(f"[TDOMInit] {module_name} missing sample: {missing[:5]}")
            if unexpected:
                print(f"[TDOMInit] {module_name} unexpected sample: {unexpected[:5]}")

    def freeze_encoder_neck(self):
        for module_name in ("encoder", "neck"):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            module.requires_grad_(False)
            module.eval()
        print("[FeatureFreeze] encoder and neck are frozen.")

    def freeze_tdom_branch_modules(self):
        for module_name in ("encoder", "neck", "decoder", "conv"):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            module.requires_grad_(False)
            module.eval()
        print("[TDOMFreeze] encoder, neck, decoder and conv are frozen; only fusion modules remain trainable.")

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "tail_logit_correction_enabled", False):
            for module_name in ("encoder", "neck", "decoder", "conv", "cross_attn_fusion"):
                module = getattr(self, module_name, None)
                if module is not None:
                    module.eval()
            for module in self._logit_tail_trainable_modules():
                module.train(mode)
        if getattr(self, "freeze_feature_extractor", False):
            for module_name in ("encoder", "neck"):
                module = getattr(self, module_name, None)
                if module is not None:
                    module.eval()
        if getattr(self, "freeze_tdom_branch", False):
            for module_name in ("encoder", "neck", "decoder", "conv"):
                module = getattr(self, module_name, None)
                if module is not None:
                    module.eval()
        return self

    def forward(
        self,
        x,
        cam_features=None,
        cam_exist=None,
        return_features=False,
        output_attn=False,
        return_tdom_pred=False,
        return_fusion_aux=False,
        names=None,
        save_dir=None,
        anchor_features=None,
        anchor_exist=None,
        tail_features=None,
        tail_exist=None,
        tail_feature_chunks=None,
        tail_topk_eval_choices=None,
        tail_multi_q_eval_choices=None,
        return_tail_off_pred=False,
    ):
        """
        Args:
            x: (image [B,3,H,W], dsm [B,1,H,W])
            cam_features: list[N_cam], 每个 dict{res2/3/4/5: [B,C,H,W]}
                          如果为 None 则不做融合
            cam_exist: Tensor[B, N_cam] bool
            anchor_features / tail_features: optional separate view lists for
                ``fusion_type: sail_image_memory``.  Existing callers can keep
                passing cam_features only; it is then used as the anchor list.
            return_features: bool, 是否返回中间特征 (TDOM 和 融合后特征)
            output_attn: bool, 是否返回注意力权重
            names: list[str], 切片名称，用于保存特征
            save_dir: str, 特征保存根目录
        """
        if return_tail_off_pred:
            if (not self._uses_sail_image_memory
                    or self.tail_logit_correction_enabled
                    or self.cross_attn_fusion.memory_output_mode != "anchor_plus_tail"
                    or (anchor_features is None and cam_features is None)
                    or any(value is not None for value in (
                        tail_feature_chunks, tail_topk_eval_choices,
                        tail_multi_q_eval_choices))
                    or output_attn or return_tdom_pred or return_features):
                raise ValueError(
                    "tail-off supervision requires dense SAIL anchor_plus_tail "
                    "fusion and cannot combine with streaming/other prediction returns"
                )
        image, dsm = x

        # if add_lidar:
        #     coords = dsm_to_points(dsm)
        #     point_dict = {}
        #     point_dict['coords'] = coords
        #     point_dict['feat'] = torch.zeros_like(coords)

        #     grid_sampler = GridSample(
        #         grid_size=0.8,                # 体素大小 
        #         hash_type="fnv",              # 哈希算法类型
        #         mode="train",                 # 当前为训练模式
        #         return_grid_coord=True, 
        #     )
        #     point_dict['grid_coords'] = grid_sampler(point_dict)

        #     B = len(coords)
        #     lengths = [len(coords[i]) for i in range(B)]
        #     batch_idx = []
        #     for i in range(B):
        #         batch_idx.append(torch.full((lengths[i],), i%B, dtype=torch.long))
        #     batch_idx = torch.cat(batch_idx, dim=0)
        #     point_dict['batch'] = batch_idx

        #     point_feature = self.point_transformer(point_dict)
        #     H, W = dsm.shape[-2:]
        #     C = point_feature.shape[-1]
        #     point_feat2d = point_feature.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        #     # 构造多尺度特征
            

        _, vit_features = self.encoder(image)

        spatial_input = build_spatial_input(self.neck, image, dsm)

        if self.neck is not None:
            neck_out = self.neck((spatial_input, vit_features))
        else:
            neck_out = vit_features

        # neck_out: list[4] of [B, C, H, W]  (res2~res5 顺序)
        # 如果是 dict，转成 list
        if isinstance(neck_out, dict):
            scale_keys = ["res2", "res3", "res4", "res5"]
            tdom_features = [neck_out[k] for k in scale_keys if k in neck_out]
        else:
            tdom_features = list(neck_out)

        if self.tail_logit_correction_enabled:
            if any(value is not None for value in (
                    tail_feature_chunks, tail_topk_eval_choices, tail_multi_q_eval_choices)):
                raise ValueError("Local logit correction first version supports dense fixed-Q only, not streaming/multi-Q")
            anchors = anchor_features if anchor_features is not None else cam_features
            exists = anchor_exist if anchor_exist is not None else cam_exist
            out, fusion_aux = self._forward_tail_logit_correction(
                tdom_features, image, anchors, exists, tail_features, tail_exist)
            extra = []
            if return_tdom_pred:
                with torch.no_grad():
                    extra.append(self.conv(upsample_x_to_fit_y(self.decoder(tdom_features), image)))
            if output_attn:
                extra.append(None)
            if return_fusion_aux:
                extra.append(fusion_aux)
            return (out, *extra) if extra else out

        # Cross-Attention Fusion
        attns = None
        fusion_aux = None
        # Only explicitly opted-in layers consume separate anchor/tail inputs.
        # Legacy camera-fusion interfaces stay unchanged.
        sail_inputs = {}
        if self._uses_sail_image_memory or self._uses_tdom_joint_view_attention:
            sail_inputs = {
                "anchor_features": anchor_features,
                "anchor_exist": anchor_exist,
                "tail_features": tail_features,
                "tail_exist": tail_exist,
            }
        if tail_feature_chunks is not None:
            if not self._uses_sail_image_memory:
                raise ValueError("tail_feature_chunks is only supported by sail_image_memory fusion")
            if anchor_features is None:
                raise ValueError("All-tail SAIL inference requires anchor_features")
            if tail_topk_eval_choices is not None:
                if output_attn or return_fusion_aux or return_tdom_pred:
                    raise ValueError(
                        "multi-TopK prediction does not return attention/auxiliary outputs"
                    )
                fused_features = self.cross_attn_fusion.forward_streaming_multi_topk(
                    tdom_features,
                    anchor_features=anchor_features,
                    anchor_exist=anchor_exist,
                    tail_feature_chunks=tail_feature_chunks,
                    topk_values=tail_topk_eval_choices,
                )
            elif output_attn and return_fusion_aux:
                fused_features, attns, fusion_aux = self.cross_attn_fusion.forward_streaming(
                    tdom_features,
                    anchor_features=anchor_features,
                    anchor_exist=anchor_exist,
                    tail_feature_chunks=tail_feature_chunks,
                    output_attn=True,
                    return_aux=True,
                )
            elif output_attn:
                fused_features, attns = self.cross_attn_fusion.forward_streaming(
                    tdom_features,
                    anchor_features=anchor_features,
                    anchor_exist=anchor_exist,
                    tail_feature_chunks=tail_feature_chunks,
                    output_attn=True,
                )
            elif return_fusion_aux:
                fused_features, fusion_aux = self.cross_attn_fusion.forward_streaming(
                    tdom_features,
                    anchor_features=anchor_features,
                    anchor_exist=anchor_exist,
                    tail_feature_chunks=tail_feature_chunks,
                    return_aux=True,
                )
            else:
                fused_features = self.cross_attn_fusion.forward_streaming(
                    tdom_features,
                    anchor_features=anchor_features,
                    anchor_exist=anchor_exist,
                    tail_feature_chunks=tail_feature_chunks,
                )
        elif tail_multi_q_eval_choices is not None:
            if not self._uses_sail_image_memory:
                raise ValueError(
                    "tail_multi_q_eval_choices is only supported by sail_image_memory fusion"
                )
            if tail_features is None:
                raise ValueError(
                    "tail_multi_q_eval_choices requires loaded tail_features"
                )
            fused_features = self.cross_attn_fusion.forward_multi_fixed_q(
                tdom_features,
                anchor_features=anchor_features,
                anchor_exist=anchor_exist,
                tail_features=tail_features,
                tail_exist=tail_exist,
                q_values=tail_multi_q_eval_choices,
            )
        elif cam_features is not None or anchor_features is not None:
            if output_attn and return_fusion_aux:
                fused_features, attns, fusion_aux = self.cross_attn_fusion(
                    tdom_features,
                    cam_features,
                    cam_exist,
                    output_attn=True,
                    return_aux=True,
                    **sail_inputs,
                )
            elif output_attn:
                fused_features, attns = self.cross_attn_fusion(
                    tdom_features, cam_features, cam_exist, output_attn=True,
                    **sail_inputs,
                )
            elif return_fusion_aux or return_tail_off_pred:
                fused_features, fusion_aux = self.cross_attn_fusion(
                    tdom_features,
                    cam_features,
                    cam_exist,
                    return_aux=True,
                    **sail_inputs,
                )
            else:
                fused_features = self.cross_attn_fusion(
                    tdom_features, cam_features, cam_exist, **sail_inputs
                )
        else:
            fused_features = tdom_features

        if return_tail_off_pred:
            # Reuse the SAME differentiable TDOM and anchor memory. No second
            # backbone/neck/R/tail readout pass, no detach, no new parameters.
            tail_off_features = [
                self.cross_attn_fusion._fuse_tdom_memory(tdom, memory, sid)
                for sid, (tdom, memory) in enumerate(zip(
                    tdom_features, fusion_aux["anchor_memory"]))
            ]
            if self.training:
                # Update BatchNorm once on both states, rather than twice in
                # order-dependent on/off passes. Dropout remains unchanged.
                paired_features = [torch.cat([on, off], dim=0)
                                   for on, off in zip(fused_features, tail_off_features)]
                decoded = self.decoder(paired_features)
                logits = self.conv(upsample_x_to_fit_y(decoded, image))
                out, tail_off_out = logits.split(image.shape[0], dim=0)
            else:
                # Preserve the exact ordinary ON eval batch shape/semantics.
                out = self.conv(upsample_x_to_fit_y(self.decoder(fused_features), image))
                tail_off_out = self.conv(upsample_x_to_fit_y(self.decoder(tail_off_features), image))
            if return_fusion_aux:
                return out, tail_off_out, fusion_aux
            return out, tail_off_out

        if isinstance(fused_features, dict):
            outputs = {}
            for topk, topk_features in fused_features.items():
                final_feature = self.decoder(topk_features)
                topk_out = upsample_x_to_fit_y(final_feature, image)
                outputs[int(topk)] = self.conv(topk_out)
            return outputs

        final_feature = self.decoder(fused_features)
        out = upsample_x_to_fit_y(final_feature, image)
        out = self.conv(out)

        tdom_out = None
        if return_tdom_pred:
            with torch.no_grad():
                tdom_feature = self.decoder(tdom_features)
                tdom_out = upsample_x_to_fit_y(tdom_feature, image)
                tdom_out = self.conv(tdom_out)

        if output_attn and return_tdom_pred and return_fusion_aux:
            return out, tdom_out, attns, fusion_aux
        if output_attn and return_tdom_pred:
            return out, tdom_out, attns
        if return_tdom_pred and return_fusion_aux:
            return out, tdom_out, fusion_aux
        if output_attn and return_fusion_aux:
            return out, attns, fusion_aux
        if output_attn:
            return out, attns
        if return_tdom_pred:
            return out, tdom_out
        if return_fusion_aux:
            return out, fusion_aux

        return out


# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder, neck=None, out_class=1, **kwargs):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         lora_config = LoraConfig(
#             r=8,                   # Rank: 8 就够了，性价比最高
#             lora_alpha=16,         # Alpha: 通常是 Rank 的 2 倍
            
#             # 【关键】这里填你要微调的层
#             # 这里的名字是基于 self.encoder.backbone 内部的模块名匹配的
#             target_modules=["qkv"], 
            
#             lora_dropout=0.1,      # 防止过拟合
#             bias="none",
            
#             # 如果你有 Adapter 层 (比如 'spatial_prior', 'interaction')
#             # 你希望这些层全量训练，而不是 LoRA 微调，就把它们的名字加到这里：
#             # modules_to_save=["adapter", "spatial_prior", "interaction"],
#         )
#         # 把 LoRA 注入到 backbone 中
#         # 注意：这里我们只包 backbone，保留外面的 wrapper 结构
#         self.encoder.backbone = get_peft_model(self.encoder.backbone, lora_config)
        
#         # 打印一下看看有多少参数是活的
#         self.encoder.backbone.print_trainable_parameters()
#         self.neck = neck
#         self.decoder = decoder
#         self.out_class = out_class
        
#     def forward(self, x, targets=None):
#         # 1. 拆包输入
#         image, dsm = x
        
#         # 2. Encoder 前向
#         # 假设 encoder 返回 (cls_token, features_list)
#         # 如果你的 encoder 只返回 features，这里要改一下
#         encoder_outputs = self.encoder(image)
#         if isinstance(encoder_outputs, tuple):
#             _, vit_features = encoder_outputs
#         else:
#             vit_features = encoder_outputs

#         # 3. Neck 前向
#         # 注意：这里你保留了 spatial_input，说明你用的是复杂的 SPM Neck
#         # 确保你的 Neck 初始化时 in_channels=4 (RGB+DSM)
#         spatial_input = torch.cat([image, dsm], dim=1) 
#         features = self.neck((spatial_input, vit_features))

#         # 4. 特征对齐 (关键修改：强制重命名 Key)
#         # SOLOv2Decoder 需要 p2, p3, p4, p5
#         target_keys = ["p2", "p3", "p4", "p5"]
#         feature_dict = {}

#         if isinstance(features, (list, tuple)):
#             # 列表按顺序映射
#             for k, v in zip(target_keys, features):
#                 feature_dict[k] = v
#         elif isinstance(features, dict):
#             # 字典按值排序映射，或者尝试重命名
#             # 方案 A: 如果知道 key 是 res2...res5，直接取
#             if "res2" in features:
#                 feature_dict["p2"] = features["res2"]
#                 feature_dict["p3"] = features["res3"]
#                 feature_dict["p4"] = features["res4"]
#                 feature_dict["p5"] = features["res5"]
#             else:
#                 # 方案 B: 无论 Key 是什么，按 Key 排序后强制赋值给 p2-p5
#                 # 这能防止 Key 名称不匹配的问题
#                 sorted_keys = sorted(features.keys())
#                 for i, k in enumerate(target_keys):
#                     if i < len(sorted_keys):
#                         feature_dict[k] = features[sorted_keys[i]]
#         else:
#             raise TypeError(f"Neck output type {type(features)} not supported.")

#         # 5. 训练/推理分支
#         if self.training:
#             if targets is None:
#                 raise ValueError("In training mode, 'targets' (gt_instances) must be provided.")
            
#             # 返回 Loss 字典
#             return self.decoder(feature_dict, gt_instances=targets)
        
#         else:
#             # 推理模式
#             image_sizes = [(image.shape[2], image.shape[3])] * image.shape[0]
#             # 返回 Results List
#             return self.decoder(feature_dict, image_sizes=image_sizes)


# mask2former
# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder, out_class, neck=None,pixel_decoder=None, **kwargs):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         lora_config = LoraConfig(
#             r=16,                   # Rank: 8 就够了，性价比最高
#             lora_alpha=32,         # Alpha: 通常是 Rank 的 2 倍
            
#             # 【关键】这里填你要微调的层
#             # 这里的名字是基于 self.encoder.backbone 内部的模块名匹配的
#             target_modules=["qkv"], 
            
#             lora_dropout=0.1,      # 防止过拟合
#             bias="none",
            
#             # 如果你有 Adapter 层 (比如 'spatial_prior', 'interaction')
#             # 你希望这些层全量训练，而不是 LoRA 微调，就把它们的名字加到这里：
#             # modules_to_save=["adapter", "spatial_prior", "interaction"],
#         )
#         # 把 LoRA 注入到 backbone 中
#         # 注意：这里我们只包 backbone，保留外面的 wrapper 结构
#         self.encoder.backbone = get_peft_model(self.encoder.backbone, lora_config)
#         self.encoder.backbone.print_trainable_parameters()
#         self.neck = neck
#         self.pixel_decoder = pixel_decoder
#         self.decoder = decoder
#         self.out_class = out_class
        

#     def forward(self, x, proj_feature=None):
#         image, dsm = x
#         _, vit_features = self.encoder(image)
#         spatial_input = torch.cat([image, dsm], dim=1) 
#         features_dict = self.neck((spatial_input, vit_features))
#         # mask_features, _, multi_scale_features = self.pixel_decoder(features_dict)
#     #     vis_pca_feature(
#     #     mask_features,
#     #     title="mask_features_PCA",
#     #     save_dir="/path/to/project/experiment/MVMapper/Hangzhou_XYZ_DSM_newinstencedata_v2_testdata6_1024_dinov3_pixel_decoder_mask2former_Lora_lr1e-4/debug_pca"
#     # )

#     #     for i, feat in enumerate(multi_scale_features):
#     #         vis_pca_feature(
#     #             feat,
#     #             title=f"multi_scale_features_{i}_PCA",
#     #             save_dir="/path/to/project/experiment/MVMapper/Hangzhou_XYZ_DSM_newinstencedata_v2_testdata6_1024_dinov3_pixel_decoder_mask2former_Lora_lr1e-4/debug_pca1"
#     #         )

#         # outputs = self.decoder(x=multi_scale_features, mask_features=mask_features)
        
#         return features_dict


class SegmentationModel(nn.Module):
    def __init__(self, model, **kwargs):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.intermediate_hook = None
        self.intermediate = kwargs.get('intermediate',None)
        if 'intermediate' in kwargs and kwargs.get('register_hook',False):
            if isinstance(model, EncoderDecoder):
                # encoder-decoder模型强制在encoder的输出进行融合，这样不需要改变原有模型的forward
                self.intermediate_hook = self.registerforintermediate_Encoder_Decoder('encoder')
            else:
                # 其他模型需要更改forward进行融合
                self.intermediate_hook = self.registerforintermediate(kwargs['intermediate'])

    def registerforintermediate(self, intermediate):
        return HookClass(self.model,intermediate)

    def registerforintermediate_Encoder_Decoder(self, intermediate):
        return HookClass(self.model, intermediate)
        # for name, module in self.model.encoder.named_modules():
        #     if name == intermediate:
        #         return HookClass(self.model.encoder,intermediate)
        # return HookClass(self.model.decoder,intermediate)
    def forward(self, x, intermediate_input = None):
        return self.model(x) if intermediate_input is None else self.model(x,intermediate_input)

class MVMapper(nn.Module):
    def __init__(self,
                 image_branch,
                 spatial_fusion,
                 xyz_branch,
                 connect_cal='cat'):
        super(MVMapper, self).__init__()
        self.image_branch = image_branch
        self.spatial_fusion = spatial_fusion
        self.xyz_branch = xyz_branch
        # self.fusion_module = FusionModule()
        self.connect_cal=connect_cal
        self.stage = 'stage1'

    def img_forward(self, x):
        img_feature=self.image_branch(x)
        intermediate_feature = self.image_branch.intermediate_hook.out_features
        return img_feature, intermediate_feature
    def set_stage(self,stage):
        self.stage = stage
    def xyz_forward(self,x, intermediate_feature):
        # 这个方案行不通
        # for name,module in self.xyz_branch.named_modules():
        #     x = module(x)
        #     if name==self.self.xyz_branch.intermediate:
        #         x=self.fusion_module(x,intermediate_feature)
        # return x
        pass

    def train_forward(self, x):
        grid = x["grid"]
        batch_info = x["batch_info"]
        img_x, xyz_x = x["image"]
        img_feature, intermediate_feature = self.img_forward(img_x)
        proj_feature = self.spatial_fusion(intermediate_feature, grid, batch_info)
        xyz_feature = self.xyz_branch(xyz_x, proj_feature)
        return img_feature, xyz_feature

    @torch.no_grad()
    def test_forwad(self,x):
        if self.stage == 'stage1':
            return self.img_forward(x)[0]
        else:
            return self.train_forward(x)[1]

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.test_forwad(x)



if __name__ == '__main__':
    x=torch.ones([5,4,512,512])
    xyz=torch.ones([2,4,512,512])
    grid=torch.ones([5,128,128,2])
    batch_info=[3,2]
    encoder = SwinTransformer(pretrain_img_size=224,in_chans=4,window_size=7)
    encoder2 = SwinTransformer(pretrain_img_size=224,in_chans=4,window_size=7)

    in_channels=(96,192,384,768)
    # in_channels=(192, 384, 768, 1536)
    channels=512
    pool_scales=(1, 2, 3, 6)
    conv_cfg=dict(type='Conv2d')
    norm_cfg=dict(type='BatchNorm2d')
    act_cfg=dict(type='ReLU',inplace=True)
    align_corners = False
    decoder = UPerHead(in_channels,channels,pool_scales,conv_cfg,norm_cfg,act_cfg,align_corners)
    decoder2 = UPerHead(in_channels,channels,pool_scales,conv_cfg,norm_cfg,act_cfg,align_corners)

    img_branch = SegmentationModel(EncoderDecoder(encoder,decoder,1),intermediate='encoder',register_hook=True)
    xyz_branch = SegmentationModel(EncoderDecoder(encoder2,decoder2,1),intermediate='encoder')
    spatial_fusion = SpatialFusion2DV2((512, 256, 128), resize_feature=True)
    # for name ,_ in img_branch.model.encoder.named_modules():
    #     print(name)
    model = MVMapper(
        image_branch=img_branch,
        spatial_fusion = spatial_fusion,
        xyz_branch=xyz_branch)
    # print(model)
    model.eval()
    # y=model({'image':[x,xyz],'grid':grid,'batch_info':batch_info})
    img = model(x)
    model.set_stage('stage2')
    xyz = model({'image':[x,xyz],'grid':grid,'batch_info':batch_info})
    print(img.shape,xyz.shape)
