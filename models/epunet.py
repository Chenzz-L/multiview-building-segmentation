r"""
Deep building footprint update network: A semi-supervised method for
updating existing building footprint from bi-temporal remote sensing images
modified from https://github.com/HaonanGuo/SG-EPUNet
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models
from models.unet import Interp

__all__ = ['EPUNet']


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 n_filters=256,
                 kernel_size=3,
                 is_deconv=False,
                 nonlinearity=nn.LeakyReLU,
                 ):
        super().__init__()
        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0
        else:
            raise Exception('kernel_size must be 1 or 3')
        mid_channels = in_channels // 4
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size,
                               padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv:
            self.deconv2 = nn.ConvTranspose2d(mid_channels, mid_channels, 3,
                                              stride=2, padding=1,
                                              output_padding=conv_padding, bias=False)
        else:
            self.deconv2 = Interp(scale_factor=2)

        self.norm2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(mid_channels, n_filters, kernel_size,
                               padding=conv_padding, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.functional.leaky_relu
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Attetion(nn.Module):
    def __init__(self, inchannels, kernel_size=7):
        super(Attetion, self).__init__()
        self.CA = ChannelAttention(in_planes=inchannels)
        self.SA = SpatialAttention(kernel_size=kernel_size)

    def forward(self, inputs):
        out = self.CA(inputs) * inputs
        out = self.SA(out)
        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=64, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class EPUNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 resnet=torchvision.models.resnet34(weights=None)
                 ):
        super().__init__()
        filters = [64, 128, 256, 512]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention4 = Attetion(filters[2])
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention3 = Attetion(filters[1])
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention2 = Attetion(filters[0])
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention1 = Attetion(filters[0])
        self.finalconv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, out_channels, 1))
        self.midconv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(32, out_channels, 1))
        self.res4 = torchvision.models.resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1)
        self.res3 = torchvision.models.resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = torchvision.models.resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res1 = torchvision.models.resnet.BasicBlock(8, 8, stride=1, downsample=None)
        self.conv1 = nn.Conv2d(8, 1, kernel_size=1)
        self.psp = PSPModule(features=8 + 64)

    def forward(self, x, get_mid=False):
        inp_size = x.shape[-2:]
        # stem
        x = self.firstconv(x)
        x_ = self.maxpool(x)
        # Encoder
        e1 = self.layer1(x_)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        center = self.center(e4)
        # decoder
        d4 = self.decoder4(torch.cat([center, e3], 1))
        edge = self.res4(x) * F.interpolate(self.attention4(d4), x.shape[-2:], mode='bilinear', align_corners=True)
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        edge = self.res3(self.conv4(edge)) * F.interpolate(self.attention3(d3), x.shape[-2:], mode='bilinear',
                                                           align_corners=True)
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        edge = self.res2(self.conv3(edge)) * F.interpolate(self.attention2(d2), x.shape[-2:], mode='bilinear',
                                                           align_corners=True)
        d1 = self.decoder1(torch.cat([d2, x], 1))
        edge = self.res1(self.conv2(edge)) * F.interpolate(self.attention1(d1), x.shape[-2:], mode='bilinear',
                                                           align_corners=True)
        edge_result = F.interpolate(self.conv1(edge), inp_size, mode='bilinear', align_corners=True)
        f = self.psp(torch.cat([d1, F.interpolate(edge, inp_size, mode='bilinear', align_corners=True)], 1))
        mid = f
        seg_result = self.finalconv(f)

        if get_mid:
            return seg_result, edge_result, mid
        else:
            return seg_result, edge_result


class EPUNet_Decoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 resnet=torchvision.models.resnet34(weights=None),
                 ):
        super().__init__()
        filters = [64, 128, 256, 512]
        self.in_channels = in_channels
        self.out_channels = out_channels
        #
        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        # edge
        self.attention4 = Attetion(filters[2])
        self.attention3 = Attetion(filters[1])
        self.attention2 = Attetion(filters[0])
        self.attention1 = Attetion(filters[0])
        self.res4 = torchvision.models.resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1)
        self.res3 = torchvision.models.resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = torchvision.models.resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res1 = torchvision.models.resnet.BasicBlock(8, 8, stride=1, downsample=None)
        self.conv1 = nn.Conv2d(8, 1, kernel_size=1)
        self.psp = PSPModule(features=8 + 64)
        self.finalconv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, out_channels, 1))

    def forward(self, x, get_mid=False):
        inp_size = x.shape[-2:]
        # stem
        x = self.firstconv(x)
        x_ = self.maxpool(x)
        # Encoder
        e1 = self.layer1(x_)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        center = self.center(e4)
        # decoder
        d4 = self.decoder4(torch.cat([center, e3], 1))
        edge = self.res4(x) * F.interpolate(self.attention4(d4), x.shape[-2:], mode='bilinear', align_corners=True)
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        edge = self.res3(self.conv4(edge)) * F.interpolate(self.attention3(d3), x.shape[-2:], mode='bilinear',
                                                           align_corners=True)
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        edge = self.res2(self.conv3(edge)) * F.interpolate(self.attention2(d2), x.shape[-2:], mode='bilinear',
                                                           align_corners=True)
        d1 = self.decoder1(torch.cat([d2, x], 1))
        edge = self.res1(self.conv2(edge)) * F.interpolate(self.attention1(d1), x.shape[-2:], mode='bilinear',
                                                           align_corners=True)
        edge_result = F.interpolate(self.conv1(edge), inp_size, mode='bilinear', align_corners=True)
        mid = self.psp(torch.cat([d1, F.interpolate(edge, inp_size, mode='bilinear', align_corners=True)], 1))
        seg_result = self.finalconv(mid)

        if get_mid:
            return seg_result, edge_result, mid
        else:
            return seg_result, edge_result
