"""Opt-in offline fusion baselines; historical SAIL / TDOM-Q behavior is untouched."""
import torch
from torch import nn
from torch.nn import functional as F

from layers.tdom_joint_view_attention import TDOMJointViewAttentionFusion, SCALES


class OfflineMaxConcatFusion(nn.Module):
    """The full-view max is unchanged before [TDOM,max] 1x1 projection."""
    def __init__(self, embed_dim=256, num_scales=4, alpha_init=0.05):
        super().__init__()
        self.embed_dim, self.num_scales = embed_dim, num_scales
        self.final_fusion = "concat_project"
        self.mask_zero_tail_features = False
        self.tail_local_context_enabled = False
        self.final_projections = nn.ModuleList([
            nn.Conv2d(2*embed_dim, embed_dim, 1) for _ in range(num_scales)])
        with torch.no_grad():
            eye = torch.eye(embed_dim)
            for projection in self.final_projections:
                projection.weight.zero_()
                projection.weight[:, :embed_dim, 0, 0].copy_(eye)
                projection.weight[:, embed_dim:, 0, 0].copy_(alpha_init*eye)
                projection.bias.zero_()

    def forward(self, tdom_features, cam_features=None, cam_exist=None, *,
                anchor_features=None, anchor_exist=None, tail_features=None, tail_exist=None,
                output_attn=False, return_aux=False):
        features = anchor_features if anchor_features is not None else cam_features
        exists = anchor_exist if anchor_exist is not None else cam_exist
        if features is None or len(features) != 1 or tail_features:
            raise ValueError("Offline max expects one max feature and no tails")
        if len(tdom_features) != self.num_scales:
            raise ValueError("Wrong scale count")
        fused, memory = [], []
        for sid, (scale, reference) in enumerate(zip(SCALES, tdom_features)):
            value = features[0][scale].to(device=reference.device, dtype=reference.dtype)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.shape != reference.shape:
                raise ValueError("Offline max and TDOM scales must match exactly")
            if exists is not None:
                if tuple(exists.shape) != (reference.shape[0], 1):
                    raise ValueError("Max existence flags must be [B,1]")
                value = value.masked_fill(~exists.to(reference.device).bool()[:, :, None, None], 0)
            memory.append(value)
            fused.append(self.final_projections[sid](torch.cat((reference, value), 1)))
        aux = {"image_memory": memory}
        if output_attn and return_aux:
            return fused, {}, aux
        if output_attn:
            return fused, {}
        if return_aux:
            return fused, aux
        return fused


class OfflineCluster21AttentionFusion(TDOMJointViewAttentionFusion):
    """Same flat TDOM-Q attention, but zeros from real groups remain valid.

    Only empty *group slots* are masked. This deliberately is not a change
    to the default/historical raw-view attention's per-pixel zero mask.
    """
    def __init__(self, embed_dim=256, num_heads=8, num_scales=4,
                 ffn_ratio=2.0, dropout=0.0, alpha_init=0.05, eps=1e-6,
                 query_chunk_size=512):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, num_scales=num_scales,
                         ffn_ratio=ffn_ratio, dropout=dropout, alpha_init=alpha_init,
                         eps=eps, query_chunk_size=query_chunk_size,
                         tail_local_context_enabled=False, tail_local_context_scales=())
        self.mask_zero_tail_features = False

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
            if value.shape != (b, c, h, w):
                raise ValueError("Offline group and TDOM scales must match exactly")
            values.append(value)
        stacked = torch.stack(values, 1)
        if exists is None:
            flags = torch.ones((b, len(values)), device=reference.device, dtype=torch.bool)
        else:
            if tuple(exists.shape) != (b, len(values)):
                raise ValueError("Group-existence mask does not match inputs")
            flags = exists.to(reference.device).bool()
        valid = flags[:, :, None, None].expand(b, len(values), h, w)
        return stacked.masked_fill(~valid.unsqueeze(2), 0), valid
