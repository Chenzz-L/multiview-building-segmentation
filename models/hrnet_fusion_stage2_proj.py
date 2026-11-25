# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by RainbowSecret (yhyuan@pku.edu.cn)
# ------------------------------------------------------------------------------
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 
import sys
sys.path.append('/path/to/user/NAS/gwx/code/seg_framer_v2')

from layers.spatial_fusion import SpatialFusion2DV2
from models.hrnet_config import MODEL_CONFIGS, HRNET_CFG, STAGE_CFG
from models.hrnet import HighResolutionModule,blocks_dict
__all__ = ['hrnet18_fusion_stage2', 'hrnet32_fusion_stage2', 'hrnet48_fusion_stage2']

ALIGN_CORNERS = True

class HighResolutionNet(nn.Module):

    def __init__(self, cfg: HRNET_CFG, in_channels=3, out_channels=3, norm_layer=None,out_branch=(0,1)):
        super(HighResolutionNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_layer = norm_layer
        self.base_width = 64
        self.out_branch=out_branch
        # stem network
        # stem net
        self.conv1 = nn.Conv2d(self.in_channels, self.base_width, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = self.norm_layer(self.base_width)
        self.conv2 = nn.Conv2d(self.base_width, self.base_width, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = self.norm_layer(self.base_width)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = cfg.STAGE1
        num_channels = self.stage1_cfg.NUM_CHANNELS[0]
        block = blocks_dict[self.stage1_cfg.BLOCK]
        num_blocks = self.stage1_cfg.NUM_BLOCKS[0]
        self.layer1 = self._make_layer(block, self.base_width, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # stage 2
        self.stage2_cfg = cfg.STAGE2
        num_channels = self.stage2_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage2_cfg.BLOCK]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = cfg.STAGE3
        num_channels = self.stage3_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage3_cfg.BLOCK]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        
        
        # stage 4
        self.stage4_cfg = cfg.STAGE4
        num_channels = self.stage4_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage4_cfg.BLOCK]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            self.norm_layer(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=self.out_channels,
                kernel_size=cfg.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if cfg.FINAL_CONV_KERNEL == 3 else 0)
        )

        self.proj=SpatialFusion2DV2(num_channels,resize_feature=True)
        self.xyz_transition3=copy.deepcopy(self.transition3)
        self.xyz_stage3=copy.deepcopy(self.stage3)
        self.xyz_stage4=copy.deepcopy(self.stage4)
        self.xyz_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            self.norm_layer(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=1,
                kernel_size=cfg.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if cfg.FINAL_CONV_KERNEL == 3 else 0)
        )


        # self.final = nn.Sigmoid() if self.out_channels == 1 else nn.Softmax(dim=1)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        self.norm_layer(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config: STAGE_CFG, num_inchannels, multi_scale_output=True):
        num_modules = layer_config.NUM_MODULES
        num_branches = layer_config.NUM_BRANCHES
        num_blocks = layer_config.NUM_BLOCKS
        num_channels = layer_config.NUM_CHANNELS
        block = blocks_dict[layer_config.BLOCK]
        fuse_method = layer_config.FUSE_METHOD

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_out_branch(self,out_branch):
        self.out_branch=out_branch
    
    def forward(self, x):
        if isinstance(x,dict):
            grid=x["grid"]
            batch_info=x["batch_info"]
            x=x["image"]
        else:
            grid=None
            batch_info=None
        x_shape = x.shape[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.NUM_BRANCHES):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        
        
        
        x_list = []
        for i in range(self.stage3_cfg.NUM_BRANCHES):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg.NUM_BRANCHES:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        xyz_list=x_list
        
        x_list = []
        for i in range(self.stage4_cfg.NUM_BRANCHES):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg.NUM_BRANCHES:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # image branch
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x[1]=F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x[2]=F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x[3]=F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x = torch.cat([x[0], x[1], x[2], x[3]], 1)

        x1 = self.last_layer(x)
        x1 = F.interpolate(x1, x_shape, mode='bilinear', align_corners=ALIGN_CORNERS)
        

        if grid is None:
            return x1
        else:
            # xyz branch
            x2=self.proj(xyz_list,grid,batch_info)
            x2=self.xyz_stage3(x2)
            x_list = []
            for i in range(self.stage4_cfg.NUM_BRANCHES):
                if self.xyz_transition3[i] is not None:
                    if i < self.stage3_cfg.NUM_BRANCHES:
                        x_list.append(self.xyz_transition3[i](x2[i]))
                    else:
                        x_list.append(self.xyz_transition3[i](x2[-1]))
                else:
                    x_list.append(x2[i])
            x2=self.xyz_stage4(x_list)
            
            x0_h, x0_w = x2[0].size(2), x2[0].size(3)
            x2[1]=F.interpolate(x2[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x2[2]=F.interpolate(x2[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x2[3]=F.interpolate(x2[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x2 = torch.cat([x2[0], x2[1], x2[2], x2[3]], 1)
            x2=self.xyz_branch(x2)
            x2 = F.interpolate(x2, x_shape, mode='bilinear', align_corners=ALIGN_CORNERS)

            if len(self.out_branch)==2:
                return x1,x2
            else:
                return [x1,x2][self.out_branch[0]]

def _hrnet(arch, **kwargs):
    model = HighResolutionNet(MODEL_CONFIGS[arch], **kwargs)
    return model

def hrnet18_fusion_stage2(**kwargs):
    r"""HRNet-18 model
    """
    return _hrnet('hrnet18', **kwargs)

def hrnet32_fusion_stage2(**kwargs):
    r"""HRNet-32 model
    """
    return _hrnet('hrnet32', **kwargs)

def hrnet48_fusion_stage2(**kwargs):
    r"""HRNet-48 model
    """
    return _hrnet('hrnet48', **kwargs)


if __name__=='__main__':
    model=hrnet32_fusion_stage2(in_channels=3, out_channels=1)
    x=torch.rand([10,3,512,512])
    batch_info=[3,2,3,2]
    grid=torch.rand([10,512,512,2])
    # y=model({'image':x,'grid':grid,'batch_info':batch_info})
    for name,module in model.named_buffers():
        print(name)
    print("ok")
