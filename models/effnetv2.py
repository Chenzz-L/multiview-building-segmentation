import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.effnet_config import *
from models.modules import SiLU, conv_3x3_bn_silu


__all__ = ['EffNetV2', 'EffNetV2_Encoder', 'EffNetV2_xs_Encoder', 'EffNetV2_s_Encoder',
           'EffNetV2_m_Encoder', 'EffNetV2_l_Encoder', 'EffNetV2_xl_Encoder']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self,
                 cfgs,
                 in_channels=3,
                 block=MBConv,
                 width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs
        self.channels = [cfg[1] for cfg in cfgs]  # [24, 32, 48, 64, 96, 128]
        # building first layer
        stage_inp = _make_divisible(24 * width_mult, 8)
        self.conv1 = conv_3x3_bn_silu(in_channels, stage_inp, stride=2)
        # building inverted residual blocks
        self.stages = []
        for j, cfg in enumerate(self.cfgs):
            t, c, n, s, use_se = cfg
            stage_oup = _make_divisible(c * width_mult, 8)
            layers = []
            for i in range(n):
                layers.append(block(stage_inp, stage_oup, s if i == 0 else 1, t, use_se))
                stage_inp = stage_oup
            stage_name = f'stage{j}'
            self.add_module(stage_name, nn.Sequential(*layers))
            self.stages.append(stage_name)
        # initialize_weights

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        out_features = []
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            x = stage(x)
            out_features.append(x)
            # print(x.shape)
            # if i in self.out_indices:
            #     outs.append(x)
        return out_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def EffNetV2_Encoder(net_size, in_channels):
    assert net_size in effnetv2_cfgs.keys()
    return EffNetV2(cfgs=effnetv2_cfgs[net_size], in_channels=in_channels, block=MBConv)


def EffNetV2_xs_Encoder(in_channels):
    return EffNetV2_Encoder(net_size='xs', in_channels=in_channels)


def EffNetV2_s_Encoder(in_channels):
    return EffNetV2_Encoder(net_size='s', in_channels=in_channels)


def EffNetV2_m_Encoder(in_channels):
    return EffNetV2_Encoder(net_size='m', in_channels=in_channels)


def EffNetV2_l_Encoder(in_channels):
    return EffNetV2_Encoder(net_size='l', in_channels=in_channels)


def EffNetV2_xl_Encoder(in_channels):
    return EffNetV2_Encoder(net_size='xl', in_channels=in_channels)


if __name__ == "__main__":
    print('m' in effnetv2_cfgs.keys())
    model = EffNetV2(cfgs=effnetv2_cfgs['xl']).cuda()
    # model.eval()
    x1 = torch.ones([1, 3, 384, 384]).cuda()
    # y1 = torch.ones([1, 1, 768, 768]).cuda()
    y1 = model(x1)
    # # print(out.shape)
    # print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    from torchsummary import summary
    # model = EffNetV2_xs_PSP().cuda()
    # model.eval()
    # # s = summary(model, input_size=(3, 768, 768), batch_size=1)
    # print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    # import time
    # t = time.time()
    # for kk in range(10):
    #     out = model(x1)
    # print(time.time() - t)
