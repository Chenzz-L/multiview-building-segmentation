#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Interp', 'ConvBlock', 'UNetDownx2', 'UNetDownx3', 'UNetUpx2', 'UNetUpx3', 'UNet', ]


class Interp(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        """
        Args:
            size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]) – output spatial size.
            scale_factor (float or Tuple[float]) – multiplier for spatial size.
            Has to match input size if it is a tuple.
            mode (string) – algorithm used for upsampling:
            ‘nearest’ | ‘linear’ | ‘bilinear’ | ‘trilinear’ | ‘area’. Default: ‘nearest’
            align_corners (bool, optional) – if True, the corner pixels of the input and output tensors are aligned,
            and thus preserving the values at those pixels.
            This only has effect when mode is linear, bilinear, or trilinear. Default: False
        """

        super(Interp, self).__init__()
        self.interpolate = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(ConvBlock, self).__init__()
        # convolution block
        if is_bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch), )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True), )

    def forward(self, x):
        x = self.block(x)
        return x


class UNetDownx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetDownx2, self).__init__()
        # convolution block
        if is_bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch), )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True), )

    def forward(self, x):
        x = self.block(x)
        return x


class UNetDownx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetDownx3, self).__init__()
        # convolution block
        if is_bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch), )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True), )

    def forward(self, x):
        x = self.block(x)
        return x


class UNetUpx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=False, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetUpx2, self).__init__()
        # upsampling and convolution block
        if is_deconv:
            self.upscale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        else:
            if is_bn:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.BatchNorm2d(out_ch),
                    Interp(scale_factor=2), )
            else:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    Interp(scale_factor=2), )
        self.block = UNetDownx2(in_ch, out_ch, is_bn, is_leaky, alpha)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.block(x)
        return x


class UNetUpx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=False, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetUpx3, self).__init__()
        # upsampling and convolution block
        if is_deconv:
            self.upscale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        else:
            if is_bn:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.BatchNorm2d(out_ch),
                    Interp(scale_factor=2), )
            else:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    Interp(scale_factor=2), )
        self.block = UNetDownx3(in_ch, out_ch, is_bn, is_leaky, alpha)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.block(x)
        return x


class UNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 basic_channels=64,
                 is_bn=True,
                 is_leaky=True,
                 is_deconv=False,
                 alpha=0.1):
        super(UNet, self).__init__()
        kernels = [x * basic_channels for x in [1, 2, 4, 8, 16]]
        # down&pooling
        self.downblock1 = UNetDownx2(
            in_channels, kernels[0], is_bn, is_leaky, alpha)
        self.maxpool1 = nn.MaxPool2d(2)

        self.downblock2 = UNetDownx2(
            kernels[0], kernels[1], is_bn, is_leaky, alpha)
        self.maxpool2 = nn.MaxPool2d(2)

        self.downblock3 = UNetDownx2(
            kernels[1], kernels[2], is_bn, is_leaky, alpha)
        self.maxpool3 = nn.MaxPool2d(2)

        self.downblock4 = UNetDownx3(
            kernels[2], kernels[3], is_bn, is_leaky, alpha)
        self.maxpool4 = nn.MaxPool2d(2)

        # center convolution
        self.center = ConvBlock(kernels[3], kernels[4], is_bn, is_leaky, alpha)

        # up&concating
        self.upblock4 = UNetUpx3(
            kernels[4], kernels[3], is_deconv, is_bn, is_leaky, alpha)

        self.upblock3 = UNetUpx2(
            kernels[3], kernels[2], is_deconv, is_bn, is_leaky, alpha)

        self.upblock2 = UNetUpx2(
            kernels[2], kernels[1], is_deconv, is_bn, is_leaky, alpha)

        self.upblock1 = UNetUpx2(
            kernels[1], kernels[0], is_deconv, is_bn, is_leaky, alpha)

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], out_channels, 1),
            nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1), )

    def forward(self, x):
        dx11 = self.downblock1(x)
        dx12 = self.maxpool1(dx11)

        dx21 = self.downblock2(dx12)
        dx22 = self.maxpool2(dx21)

        dx31 = self.downblock3(dx22)
        dx32 = self.maxpool3(dx31)

        dx41 = self.downblock4(dx32)
        dx42 = self.maxpool4(dx41)

        cx = self.center(dx42)

        ux4 = self.upblock4(cx, dx41)
        ux3 = self.upblock3(ux4, dx31)
        ux2 = self.upblock2(ux3, dx21)
        ux1 = self.upblock1(ux2, dx11)

        return self.outconv1(ux1)


if __name__ == "__main__":
    pass
    x_ = torch.ones([1, 3, 512, 512]).cuda()
    generator = UNet(in_channels=3, out_channels=1, basic_channels=64).cuda()
    total_params = sum(p.numel() for p in generator.parameters())
    gen_y = generator(x_)
    print("UNet->:")
    print(" Params: {:0.1f}M".format(total_params / (10 ** 6)))
    print(" Network output: ", gen_y.shape)
