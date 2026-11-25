import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import Inference, upsample_x_to_fit_y_and_add_y


__all__ = ['FPN_Decoder']


class FPN_Decoder(nn.Module):
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
        assert len(in_channels) == len(feature_indices)
        super(FPN_Decoder, self).__init__()
        self.in_channels = in_channels
        self.feature_indices = feature_indices
        self.basic_channels = basic_channels
        self.out_channels = out_channels
        self.stages = []
        for i, in_channel in enumerate(in_channels):
            stage_name = f'stage{i}'
            self.add_module(stage_name, nn.Conv2d(in_channel, basic_channels, kernel_size=1, stride=1))
            self.stages.append(stage_name)
        # Smooth layers
        self.smooth = nn.Conv2d(basic_channels, basic_channels, kernel_size=3, stride=1, padding=1)
        # Inference layers
        self.inference = Inference(inp=basic_channels, oup=out_channels, mip=basic_channels * 2, p=0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        # print('_initialize_weights')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, in_features):
        out_features = []
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            # print(in_features[self.feature_indices[i]].shape)
            if i == 0:
                out_feature = stage(in_features[self.feature_indices[i]])
            else:
                out_feature = upsample_x_to_fit_y_and_add_y(x=out_features[i-1],
                                                            y=stage(in_features[self.feature_indices[i]]))
            # print(out_feature.shape)
            out_features.append(out_feature)
        out = out_features[-1]
        out = self.smooth(out)
        out = self.inference(out)
        # print(out.shape)
        return out
