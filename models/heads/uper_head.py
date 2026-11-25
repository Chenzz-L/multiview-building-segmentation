import sys
from typing import Optional,Dict,Tuple
sys.path.append('/path/to/user/NAS/gwx/code/cnncode')
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import PPM
from layers import ConvModule

class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 pool_scales:Tuple[int]  = (1, 2, 3, 6),
                 conv_cfg:Optional[Dict] = None,
                 norm_cfg:Optional[Dict] = None,
                 act_cfg:Optional[Dict]  = dict(type='ReLU'),
                 align_corners=False):
        super(UPerHead, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv_cfg=conv_cfg
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg
        self.align_corners=align_corners

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.out_channels,
            self.out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.out_channels,
                self.out_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.out_channels,
            self.out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        # inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1]+ F.interpolate(laterals[i],
                                             size=prev_shape,
                                             mode='bilinear',
                                             align_corners=self.align_corners)
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i],
                                        size=fpn_outs[0].shape[2:],
                                        mode='bilinear',
                                        align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        return output

if __name__=='__main__':
        in_channels=(96,192,384,768)
        out_channels=512
        pool_scales=(1, 2, 3, 6)
        conv_cfg=dict(type='Conv2d')
        norm_cfg=dict(type='BatchNorm2d')
        act_cfg=dict(type='ReLU',inplace=True)
        align_corners=False
        
        model=UPerHead(in_channels,out_channels,pool_scales,conv_cfg,norm_cfg,act_cfg,align_corners)
        print(model)
