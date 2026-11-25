#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import torch
import torch.nn as nn
from models.unet import *

__all__ = ['BRNet', 'BRNetv2', 'UNet']


class ConvUnit(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, is_bn, is_leaky, alpha):
        super(ConvUnit, self).__init__()
        # convolution block
        elems = [nn.Conv2d(in_ch, out_ch, kernel), nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True)]
        if is_bn:
            elems.append(nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*elems)

    def forward(self, x):
        x = self.block(x)
        return x


class UpPredict(nn.Module):
    def __init__(self, in_ch, out_ch, scale, is_bn, is_deconv, is_leaky, alpha):
        super(UpPredict, self).__init__()
        # upsampling and convolution block
        # H_out=(H−1)∗stride[0]−2∗padding[0]+kernel_size[0]+output_padding[0]
        elems = []
        if is_deconv:
            elems.append(nn.ConvTranspose2d(in_ch, out_ch, 2,
                                            stride=scale, output_padding=scale - 2))
        else:
            elems.append(nn.Upsample(scale_factor=scale))
            elems.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            elems.append(nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True))
        if is_bn:
            elems.append(nn.BatchNorm2d(out_ch))

        self.uppredict = nn.Sequential(*elems)

    def forward(self, x):
        x = self.uppredict(x)
        return x


class Backend(nn.Module):
    def __init__(self,
                 in_channels=3,
                 basic_channels=64,
                 is_deconv=False,
                 is_bn=False,
                 is_leaky=False,
                 alpha=0.1, ):
        super(Backend, self).__init__()
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

        return ux4, ux3, ux2, ux1


class BRNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 basic_channels=64,
                 is_bn=True,
                 is_leaky=True,
                 is_deconv=False,
                 alpha=0.1):
        super(BRNet, self).__init__()
        kernels = [x * basic_channels for x in [1, 2, 4, 8, 16]]
        self.backend = Backend(in_channels=in_channels,
                               basic_channels=basic_channels,
                               is_deconv=is_deconv,
                               is_bn=is_bn,
                               is_leaky=is_leaky,
                               alpha=alpha)

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], out_channels, 1),
            nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1), )

        self.outconv2 = nn.Sequential(
            nn.Conv2d(kernels[0], out_channels, 1),
            nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1), )

    def forward(self, x):
        ux4, ux3, ux2, ux1 = self.backend(x)
        out_1 = self.outconv1(ux1)
        out_2 = self.outconv2(ux1)
        return out_1, out_2


class BRNetv2(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 basic_channels=64,
                 is_bn=True,
                 is_leaky=True,
                 is_deconv=False,
                 alpha=0.1):
        super(BRNetv2, self).__init__()
        self.out_channels = out_channels
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

        # branch for segmentation
        self.upblock4 = UNetUpx3(
            kernels[4], kernels[3], is_deconv, is_bn, is_leaky, alpha)
        self.upblock3 = UNetUpx2(
            kernels[3], kernels[2], is_deconv, is_bn, is_leaky, alpha)
        self.upblock2 = UNetUpx2(
            kernels[2], kernels[1], is_deconv, is_bn, is_leaky, alpha)
        self.upblock1 = UNetUpx2(
            kernels[1], kernels[0], is_deconv, is_bn, is_leaky, alpha)

        # branch for outline
        self.upscale_x2 = nn.ConvTranspose2d(
            kernels[4], kernels[3], 2, stride=2, output_padding=0)
        self.upscale_x4 = nn.ConvTranspose2d(
            kernels[3], kernels[2], 2, stride=2, output_padding=0)
        self.upscale_x8 = nn.ConvTranspose2d(
            kernels[2], kernels[1], 2, stride=2, output_padding=0)
        self.upscale_x16 = nn.ConvTranspose2d(
            kernels[1], kernels[0], 2, stride=2, output_padding=0)

        # generate output
        self.outconv1 = nn.Conv2d(kernels[0], out_channels, 1)
        self.outconv2 = nn.Conv2d(kernels[0], out_channels, 1)
        self.final = nn.Sigmoid() if self.out_channels == 1 else nn.Softmax(dim=1)

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
        # segmentation branch
        ux4 = self.upblock4(cx, dx41)
        ux3 = self.upblock3(ux4, dx31)
        ux2 = self.upblock2(ux3, dx21)
        ux1 = self.upblock1(ux2, dx11)
        # outline branch
        ud4 = self.upscale_x2(cx)
        ud3 = self.upscale_x4(ud4)
        ud2 = self.upscale_x8(ud3)
        ud1 = self.upscale_x16(ud2)

        seg_result = self.outconv1(ux1)  # seg
        edge_result = self.outconv2(ud1)  # edge
        return seg_result, edge_result


if __name__ == "__main__":
    # Hyper Parameters
    x_ = torch.ones([1, 3, 512, 512]).cuda()

    generator = BRNet(in_channels=3, out_channels=1).cuda()
    total_params = sum(p.numel() for p in generator.parameters())
    gen_y = generator(x_)
    print("BRNet->:")
    print(" Params: {:0.1f}M".format(total_params / (10 ** 6)))
    print(" Network output 1", gen_y[0].shape)
    print(" Network output 2", gen_y[1].shape)

    generator = BRNetv2(in_channels=3, out_channels=1).cuda()
    total_params = sum(p.numel() for p in generator.parameters())
    gen_y = generator(x_)
    print("BRNetv2->:")
    print(" Params: {:0.1f}M".format(total_params / (10 ** 6)))
    print(" Network output 1", gen_y[0].shape)
    print(" Network output 2", gen_y[1].shape)
    print(generator)
