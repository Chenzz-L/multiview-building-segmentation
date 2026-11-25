import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.modules import Inference, conv_3x3, conv_1x1, upsample_x_to_fit_y_and_add_y

__all__ = ['RefineTopDown', 'RefinePSP_Decoder']


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, sub_channels=None, sizes=(1, 2, 3, 6)):
        r"""
        pyramid pooling module
        :param in_channels: (int)
        :param out_channels: (int)
        :param sub_channels: (int) if is None, be instead by in_channels // len(sizes)
        :param sizes: (tuple)
        """
        super().__init__()
        self.in_channels = in_channels
        if sub_channels is None:
            self.sub_channels = in_channels // len(sizes)
        else:
            self.sub_channels = sub_channels
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        mid_channels = in_channels + self.sub_channels * len(sizes)
        self.out_conv = conv_1x1(mid_channels, out_channels, stride=1)
        self.relu = nn.ReLU()

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(self.in_channels, self.sub_channels, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [feats]
        for stage in self.stages:
            prior = stage(feats)
            prior = F.interpolate(input=prior, size=(h, w), mode='bicubic', align_corners=True)
            priors.append(prior)
        bottle = self.out_conv(torch.cat(priors, 1))
        return self.relu(bottle)


class RefineConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefineConv, self).__init__()
        self.conv1 = conv_3x3(inp=in_channels, oup=in_channels)
        self.conv2 = conv_3x3(inp=in_channels, oup=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x0 + x


class RefineFusion(nn.Module):
    def __init__(self, in_channels_coarse, in_channels_fine, out_channels):
        super(RefineFusion, self).__init__()
        self.block1 = conv_3x3(inp=in_channels_coarse, oup=out_channels)
        self.block2 = conv_3x3(inp=in_channels_fine, oup=out_channels)

    def forward(self, x_coarse, x_fine):
        x1 = self.block1(x_coarse)
        x2 = self.block2(x_fine)
        height, width = x2.shape[2:]
        x1_reshape = F.interpolate(x1, (height, width), mode='bicubic', align_corners=True)
        return x1_reshape + x2


class TopDownBlock(nn.Module):
    def __init__(self, top_channels, bottom_channels, out_channels, RefineConvNum=2):
        r"""
        given two input features, coarse and fine,
        first adjust two input features with same channels by conv,
        then up-sample fine feature with coarse one
        :param top_channels: (int) top layer channel with longer channels
        :param bottom_channels: (int) bottom layer channel with shorter channels
        :param out_channels: (int)
        """
        super(TopDownBlock, self).__init__()

        top_block = OrderedDict({'first_conv': conv_3x3(top_channels, out_channels), 'pre_relu': nn.ReLU()})
        bottom_block = OrderedDict({'first_conv': conv_3x3(bottom_channels, out_channels), 'pre_relu': nn.ReLU()})
        for i in range(RefineConvNum):
            top_block[f'stage{i}'] = RefineConv(out_channels, out_channels)
            bottom_block[f'stage{i}'] = RefineConv(out_channels, out_channels)
        top_block['final_conv'] = conv_3x3(out_channels, out_channels)
        bottom_block['final_conv'] = conv_3x3(out_channels, out_channels)

        self.top = nn.Sequential(top_block)
        self.bottom = nn.Sequential(bottom_block)
        # self.fuse = RefineFusion(out_channels, out_channels, out_channels)

    def forward(self, x_top, x_bottom):
        x_top = self.top(x_top)
        x_bottom = self.bottom(x_bottom)
        out = upsample_x_to_fit_y_and_add_y(x=x_top, y=x_bottom)
        return out


class RefineTopDown(nn.Module):
    def __init__(self,
                 in_channels=(512, 256, 128, 64),
                 feature_indices=(3, 2, 1, 0),
                 mid_channels=(256, 128, 64),
                 RefineConvNum=2):
        r"""
        :param in_channels: (tuple) channel num of input features
        :param feature_indices: (tuple) encoder may output many features, select which features to decode
        :param mid_channels: (tuple) output channel num, its length must be len(in_channels) - 1
        """
        assert isinstance(in_channels, (list, tuple)) and isinstance(feature_indices, (list, tuple))
        assert len(in_channels) == len(feature_indices)
        assert len(in_channels) >= 2
        assert len(mid_channels) == (len(in_channels) - 1)
        assert RefineConvNum >= 0 and isinstance(RefineConvNum, int)
        super(RefineTopDown, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.feature_indices = feature_indices
        self.stages = []
        for i in range(len(in_channels) - 1):
            stage_name = f'stage{i}'
            if i == 0:
                self.add_module(stage_name, TopDownBlock(top_channels=in_channels[i],
                                                         bottom_channels=in_channels[i + 1],
                                                         out_channels=mid_channels[i],
                                                         RefineConvNum=RefineConvNum))
            else:
                self.add_module(stage_name, TopDownBlock(top_channels=mid_channels[i - 1],  # begin with oup[i-1]
                                                         bottom_channels=in_channels[i + 1],  # merge inp[i+1]
                                                         out_channels=mid_channels[i],  # out oup[i]
                                                         RefineConvNum=RefineConvNum))
            self.stages.append(stage_name)

    def forward(self, in_features):
        out_features = []
        for i, stage_name in enumerate(self.stages):
            # print(in_features[self.feature_indices[i]].shape)
            stage = getattr(self, stage_name)
            if i == 0:
                out_feature = stage(in_features[self.feature_indices[i]],
                                    in_features[self.feature_indices[i + 1]])
            else:
                out_feature = stage(out_features[i-1],
                                    in_features[self.feature_indices[i + 1]])
            # print(out_feature.shape)
            out_features.append(out_feature)
        out = out_features[-1]
        return out


class RefinePSP_Decoder(nn.Module):
    def __init__(self,
                 in_channels=(512, 256, 128, 64),
                 feature_indices=(3, 2, 1, 0),
                 out_channels=3,
                 mid_channels=None,
                 RefineConvNum=2):
        r"""
        :param in_channels: (tuple) channel num of input features
        :param feature_indices: (tuple) encoder may output many features, select which features to decode
        :param out_channels: (int) output channel num
        :param mid_channels: (tuple) middle temporary channels
        :param RefineConvNum: (int) block num of RefineConv
        """
        super(RefinePSP_Decoder, self).__init__()
        assert len(in_channels) == len(feature_indices)
        self.in_channels = in_channels
        self.feature_indices = feature_indices
        self.out_channels = out_channels
        if mid_channels is None:
            self.mid_channels = in_channels[1::]
        else:
            self.mid_channels = mid_channels

        self.psp_top = PSPModule(in_channels=in_channels[0],
                                 out_channels=in_channels[0],
                                 sub_channels=None,
                                 sizes=(1, 2, 3, 6))
        self.topdown = RefineTopDown(in_channels=in_channels,
                                     feature_indices=feature_indices,
                                     mid_channels=self.mid_channels,
                                     RefineConvNum=RefineConvNum)
        self.psp_bottom = PSPModule(in_channels=self.mid_channels[-1],
                                    out_channels=self.mid_channels[-1],
                                    sub_channels=None,
                                    sizes=(1, 2, 3, 6))

        # inference
        self.inference = Inference(inp=self.mid_channels[-1], oup=out_channels, mip=self.mid_channels[-1] * 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, in_features):
        in_features[self.feature_indices[0]] = self.psp_top(in_features[self.feature_indices[0]])
        out_feature = self.topdown(in_features)
        out_feature = self.psp_bottom(out_feature)
        out = self.inference(out_feature)
        return out
