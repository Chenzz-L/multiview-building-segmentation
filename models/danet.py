import torch
import torch.nn as nn
from models.modules import conv_3x3_norm_act, upsample_x_to_fit_y_and_add_y, Inference
from models.fpn import FPN_Decoder
from models.refinenet import RefineTopDown

__all__ = ['DANetHead', 'FPNDA_Decoder', 'RefineDA_Decoder']


class PAM(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W)
            returns :
                out : attention value + input feature
                attention: B x (HxW) x (HxW)
        """
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        # print(proj_query.shape)
        # print(proj_key.shape)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out


class CAM(nn.Module):
    """ Channel attention module"""

    def __init__(self):
        super(CAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W)
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, channels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        r"""
        :param in_channels: (int)
        :param out_channels: (int)
        :param norm_layer:
        """
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        # pam
        self.pam_in_conv = conv_3x3_norm_act(in_channels, inter_channels, norm=norm_layer)
        self.pam = PAM(in_channels=inter_channels)
        self.pam_out_conv = conv_3x3_norm_act(inter_channels, inter_channels, norm=norm_layer)
        self.pam_conv_seg = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # cam
        self.cam_in_conv = conv_3x3_norm_act(in_channels, inter_channels, norm=norm_layer)
        self.cam = CAM()
        self.cam_out_conv = conv_3x3_norm_act(inter_channels, inter_channels, norm=norm_layer)
        self.cam_conv_seg = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # out
        self.conv_seg = Inference(inp=inter_channels, oup=out_channels, p=0.1)
        # self.conv_seg = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        # print('_initialize_weights')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # pam
        pam_feat = self.pam_in_conv(x)
        pam_feat = self.pam(pam_feat)
        pam_feat = self.pam_out_conv(pam_feat)
        pam_out = self.pam_conv_seg(pam_feat)
        # cam
        cam_feat = self.cam_in_conv(x)
        cam_feat = self.cam(cam_feat)
        cam_feat = self.cam_out_conv(cam_feat)
        cam_out = self.cam_conv_seg(cam_feat)
        # sum
        feat_sum = pam_feat + cam_feat
        pam_cam_out = self.conv_seg(feat_sum)
        return pam_cam_out, pam_out, cam_out


class FPNDA_Decoder(nn.Module):
    def __init__(self,
                 in_channels=(512, 256, 128, 64),
                 feature_indices=(3, 2, 1, 0),
                 basic_channels=64,
                 out_channels=3):
        r"""
        :param in_channels: (tuple) channel num of input features
        :param feature_indices: (tuple) encoder may output many features, select which features to decode
        :param basic_channels: (int) basic channel num of feature pyramid
        :param out_channels: (int) output channel num
        """
        super(FPNDA_Decoder, self).__init__()
        self.in_channels = in_channels
        self.feature_indices = feature_indices
        self.basic_channels = basic_channels
        self.out_channels = out_channels
        # fpn
        fpn = FPN_Decoder(in_channels=in_channels, feature_indices=feature_indices,
                          basic_channels=basic_channels, out_channels=out_channels)
        self.stages = fpn.stages
        for stage in self.stages:
            self.add_module(stage, getattr(fpn, stage))
        # da
        self.dahead = DANetHead(in_channels=basic_channels, out_channels=out_channels)

    def forward(self, in_features):
        out_features = []
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            # print(in_features[self.feature_indices[i]].shape)
            if i == 0:
                out_feature = stage(in_features[self.feature_indices[i]])
            else:
                out_feature = upsample_x_to_fit_y_and_add_y(x=out_features[i - 1],
                                                            y=stage(in_features[self.feature_indices[i]]))
            # print(out_feature.shape)
            out_features.append(out_feature)
        out = out_features[-1]
        out = self.dahead(out)
        return out[0]


class RefineDA_Decoder(nn.Module):
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
        super(RefineDA_Decoder, self).__init__()
        assert len(in_channels) == len(feature_indices)
        self.feature_indices = feature_indices
        self.out_channels = out_channels
        if mid_channels is None:
            self.mid_channels = in_channels[1::]
        else:
            self.mid_channels = mid_channels
        # top down
        self.topdown = RefineTopDown(in_channels=in_channels,
                                     feature_indices=feature_indices,
                                     mid_channels=self.mid_channels,
                                     RefineConvNum=RefineConvNum)
        # da
        self.dahead = DANetHead(in_channels=self.mid_channels[-1], out_channels=out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, in_features):
        out_feature = self.topdown(in_features)
        out = self.dahead(out_feature)
        return out[0]


if __name__ == "__main__":
    pass
    # model1 = FPNDecoder()
    model = FPNDA_Decoder()
    print(model)
    model = RefineDA_Decoder()
    print(model)
