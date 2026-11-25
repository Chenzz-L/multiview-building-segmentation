import functools
import torch
import torch.nn as nn

__all__ = ['ResNet', 'ResNet_Encoder', 'resnet_cfgs',
           'ResNet_18_Encoder', 'ResNet_34_Encoder', 'ResNet_50_Encoder', 'ResNet_101_Encoder', 'ResNet_152_Encoder']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  # 64，128，1
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 in_channels=3,
                 strides=(1, 2, 2, 2),
                 channels=(64, 128, 256, 512)):
        super(ResNet, self).__init__()
        # settings
        self.inplanes = 64
        self.strides = strides
        self.channels = [channel * block.expansion for channel in channels]
        self.downsample_factor = functools.reduce(lambda x, y: x * y, strides)
        # print(self.downsample_factor)
        # modules
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = []
        for i, layer in enumerate(layers):
            stage_name = f'layer{i+1}'
            self.add_module(stage_name, self._make_layer(block, channels[i], layer, self.strides[i]))
            self.stages.append(stage_name)
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]  # the first block stride = stride
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))  # the other block stride = 1, and downsample = None

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)  # Input Size: 3*800*800
        x = self.bn1(x)
        x = self.relu(x)  # torch.Size([1, 64, 400, 400])
        x = self.maxpool(x)  # torch.Size([1, 64, 200, 200])
        # print(x.shape)
        out_features = []
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            x = stage(x)
            # print(x.shape)
            out_features.append(x)
        return out_features


resnet_cfgs = {
    '18': (BasicBlock, (2, 2, 2, 2)),
    '34': (BasicBlock, (3, 4, 6, 3)),
    '50': (Bottleneck, (3, 4, 6, 3)),
    '101': (Bottleneck, (3, 4, 23, 3)),
    '152': (Bottleneck, (3, 8, 36, 3))
}


def ResNet_Encoder(net_size, in_channels):
    assert net_size in resnet_cfgs.keys()
    return ResNet(block=resnet_cfgs[net_size][0], layers=resnet_cfgs[net_size][1], in_channels=in_channels)

def ResNet_18_Encoder(in_channels):
    return ResNet_Encoder(net_size='18', in_channels=in_channels)


def ResNet_34_Encoder(in_channels):
    return ResNet_Encoder(net_size='34', in_channels=in_channels)


def ResNet_50_Encoder(in_channels):
    return ResNet_Encoder(net_size='50', in_channels=in_channels)


def ResNet_101_Encoder(in_channels):
    return ResNet_Encoder(net_size='101', in_channels=in_channels)


def ResNet_152_Encoder(in_channels):
    return ResNet_Encoder(net_size='152', in_channels=in_channels)


if __name__ == "__main__":
    # from torchsummary import summary
    # s = summary(generator.cuda(),input_size=(3,800,800),batch_size=1)
    # model = ResNetFPN50(input_channel=3, output_classes=1).cuda()
    # print(model)
    # if True:
    #     for param in model.conv1.parameters():
    #         param.requires_grad = False
    #     for param in model.layer1.parameters():
    #         param.requires_grad = False
    #     for param in model.layer2.parameters():
    #         param.requires_grad = False
    #     for param in model.layer3.parameters():
    #         param.requires_grad = False
    #     for param in model.layer4.parameters():
    #         param.requires_grad = False
    # print(model)

    from torchsummary import summary

    # model = ResNetFPN50(input_channel=3, output_classes=1).cuda()
    # print(resnet_settings[18])
    model = ResNet(Bottleneck, [3, 4, 23, 3], in_channels=3).cuda()
    x1 = torch.ones([1, 3, 512, 512]).cuda()
    model(x1)
    # torch.Size([1, 64, 128, 128])
    # torch.Size([1, 256, 128, 128])
    # torch.Size([1, 512, 64, 64])
    # torch.Size([1, 1024, 32, 32])
    # torch.Size([1, 2048, 16, 16])
    # model.eval()
    # t = time.time()
    # for kk in range(10):
    #     out = model(x1)
    # print(time.time() - t)
    # # s = summary(model, input_size=(3, 768, 768), batch_size=1)
    # print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    # # for kk in range(10000):
    # #     out = model(x1)
