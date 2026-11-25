import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import conv_3x3_bn_relu, upsample_x_to_fit_y

__all__ = ['DeepLabv3Plus_Decoder']


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        r"""
        Atrous Spatial Pyramid Pooling (ASPP) Module.
        :param dim_in: (int)
        :param dim_out: (int)
        :param rate: (int)
        :param bn_mom: (float)
        """
        super(ASPP, self).__init__()
        # branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        # concatenation
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [_, _, row, col] = x.size()
        # 5 branches
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # 5th branch, global avg pooling + 1x1 conv
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # concatenate 5 branches and adjust channels by 1x1 conv
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DeepLabv3Plus_Decoder(nn.Module):
    def __init__(self,
                 in_channels=(512, 64),
                 feature_indices=(3, 0),
                 basic_channels=256,
                 out_channels=3):
        r"""

        :param in_channels: (tuple) channel num of input features
        :param feature_indices: (tuple) encoder may output many features, select which features to decode
        :param basic_channels: (int) basic channel num of feature pyramid
        :param out_channels: (int) output channel num
        """
        assert len(in_channels) == len(feature_indices) == 2
        # assert len(in_channels) == len(feature_indices)
        super(DeepLabv3Plus_Decoder, self).__init__()
        self.in_channels = in_channels
        self.feature_indices = feature_indices
        self.out_channels = out_channels
        sub_channel = basic_channels // 4
        low_level_channels = in_channels[1]

        self.aspp = ASPP(dim_in=in_channels[0], dim_out=basic_channels, rate=1)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, sub_channel, 1),
            nn.BatchNorm2d(sub_channel),
            nn.ReLU(inplace=True))
        self.cat_conv = nn.Sequential(
            conv_3x3_bn_relu(sub_channel + basic_channels, basic_channels, stride=1),
            nn.Dropout(0.2),
            conv_3x3_bn_relu(basic_channels, basic_channels, stride=1),
            nn.Dropout(0.1))
        self.cls_conv = nn.Conv2d(basic_channels, out_channels, 1, stride=1)

    def forward(self, in_features):
        low_level_features = in_features[self.feature_indices[1]]
        high_level_features = in_features[self.feature_indices[0]]
        # print(low_level_features.shape, high_level_features.shape)

        high_level_features = self.aspp(high_level_features)
        low_level_features = self.shortcut_conv(low_level_features)
        high_level_features = upsample_x_to_fit_y(x=high_level_features, y=low_level_features)
        # print(low_level_features.shape, high_level_features.shape)

        out = self.cat_conv(torch.cat((high_level_features, low_level_features), dim=1))
        out = self.cls_conv(out)
        # print(out.shape)
        return out
