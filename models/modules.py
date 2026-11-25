# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ConvModule


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


def upsample_x_to_fit_y(x, y, mode='bilinear', align_corners=True):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode=mode, align_corners=align_corners)


def upsample_x_to_fit_y_and_add_y(x, y, mode='bilinear', align_corners=True):
    r"""Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
      mode: (Variable) lateral feature map.
      align_corners: (Variable) lateral feature map.
    Returns:
      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    """
    _, _, H, W = y.size()
    _, _, Hx, Wx = x.size()
    if Hx == H and Wx == W:
        x_upsample = x
    else:
        x_upsample = F.interpolate(x, size=(H, W), mode=mode, align_corners=align_corners)
    return x_upsample + y


def conv_3x3(inp, oup, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)


def conv_1x1(inp, oup, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(inp, oup, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False, dilation=1)


def conv_3x3_norm_act(inp, oup, stride=1, groups=1, dilation=1, norm=nn.BatchNorm2d, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=dilation,
                  groups=groups, bias=False, dilation=dilation),
        norm(oup),
        activation(inplace=True)
    )


def conv_3x3_bn_silu(inp, oup, stride=1, groups=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=dilation,
                  groups=groups, bias=False, dilation=dilation),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_3x3_bn_relu(inp, oup, stride=1, groups=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=dilation,
                  groups=groups, bias=False, dilation=dilation),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn_silu(inp, oup, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False, dilation=0),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_dw_bn_relu(inp, oup, k=3, s=1, p=1, b=False):
    return nn.Sequential(
        # depth wise
        nn.Conv2d(inp, inp, kernel_size=k, stride=s, padding=p, groups=inp, bias=b),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        # point wise
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=b),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def Inference(inp, oup, mip=None, p=0.1):
    r"""
    general final layer of decoder, adjust channels with final classification numbers
    :param inp: (int) input planes
    :param oup: (int) output planes
    :param mip: (int) middle planes
    :param p: (float) drop out rate
    :return:
    """
    mip = inp * 2 if mip is None else mip
    return nn.Sequential(
            conv_3x3_bn_relu(inp=inp, oup=mip, stride=1),
            nn.Dropout(p),
            conv_3x3(inp=mip, oup=oup)
        )

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out=F.interpolate(ppm_out,
                                            size=x.size()[2:],
                                            mode='bilinear',
                                            align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs
