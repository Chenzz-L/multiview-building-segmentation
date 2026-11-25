# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by RainbowSecret (yhyuan@pku.edu.cn)
# ------------------------------------------------------------------------------
import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers.spatial_fusion import SpatialFusion2DV2
from models.hrnet_config import MODEL_CONFIGS, HRNET_CFG, STAGE_CFG
from models.hrnet import HighResolutionModule,blocks_dict
from hook.basehook import HookClass

"""
    引入物方branch
"""

__all__ = ['hrnet18_image_xyz', 'hrnet32_image_xyz', 'hrnet48_image_xyz']

ALIGN_CORNERS = True

class Upsampling(nn.Module):
    def forward(self,x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x[1]=F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x[2]=F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x[3]=F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x = torch.cat([x[0], x[1], x[2], x[3]], 1)
        return x

class HighResolutionNet(nn.Module):

    def __init__(self, cfg: HRNET_CFG, in_channels=3, out_channels=3, norm_layer=None,intermediate=''):
        super(HighResolutionNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_layer = norm_layer
        self.base_width = 64
        self.intermediate=intermediate
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

        last_inp_channels = int(np.sum(pre_stage_channels))

        self.Upsampling=Upsampling()
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

    def forward(self, x):
        if isinstance(x,list):
            intermediate_input, x = x
        x_shape = x.shape[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        if self.intermediate=='layer1':
            x=x+intermediate_input

        x_list = []
        for i in range(self.stage2_cfg.NUM_BRANCHES):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        if self.intermediate=='stage2':
            y_list=[y+intermediate for intermediate,y in zip(intermediate_input,y_list)]

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
        if self.intermediate=='stage3':
            y_list=[y+intermediate for intermediate,y in zip(intermediate_input,y_list)]

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
        if self.intermediate=='stage4':
            x=[xx+intermediate for intermediate,xx in zip(intermediate_input,x)]
        # Upsampling
        x = self.Upsampling(x)
        if self.intermediate=='Upsampling':
            x=x+intermediate_input

        x = self.last_layer(x)
        x = F.interpolate(x, x_shape, mode='bilinear', align_corners=ALIGN_CORNERS)

        return x

class HighResolutionNet_image_xyz(nn.Module):
    def __init__(self, image_branch,xyz_branch,intermediate=''):
        super(HighResolutionNet_image_xyz, self).__init__()
        self.intermediate=intermediate
        self.image_branch=image_branch
        self.xyz_branch=xyz_branch
        if intermediate in ['Upsampling','layer1']:
            self.proj=SpatialFusion2DV2(128,resize_feature=True)
        else:
            self.proj=SpatialFusion2DV2([32,64,128],resize_feature=True)
        self.intermediate_hook = HookClass(self.image_branch,self.intermediate)

        self.out_branch = (0,1)

    def set_out_branch(self,out_branch):
        self.out_branch = out_branch

    def train_forward(self,x):
        grid=x["grid"]
        batch_info=x["batch_info"]
        img_x,xyz_x=x["image"]
        image_feature=self.image_branch(img_x)
        proj_feature=self.intermediate_hook.out_features
        proj_feature=self.proj(proj_feature, grid, batch_info)
        proj_feature=self.xyz_branch([proj_feature,xyz_x])
        return (image_feature,proj_feature)

    @torch.no_grad()
    def test_forwad(self,x):
        if isinstance(x,dict):
            grid=x["grid"]
            batch_info=x["batch_info"]
            img_x,xyz_x=x["image"]
        else:
            if isinstance(x,list):
                img_x,xyz_x=x
            else:
                img_x,xyz_x=x,None
            grid=None
        device = img_x.device
        
        if grid is not None:
            grid = grid.to(device)
            
        if self.out_branch==(0,) or grid==None:
            image_feature=self.image_branch(img_x)
            return image_feature
        else:
            image_feature=self.image_branch(img_x)
            proj_feature=self.intermediate_hook.out_features
            proj_feature=self.proj(proj_feature,grid,batch_info)
            proj_feature=self.xyz_branch([proj_feature,xyz_x])
            return proj_feature

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.test_forwad(x)

def _hrnet(arch, **kwargs):
    model_kwargs=kwargs.copy()
    intermediate=model_kwargs.pop('intermediate',False)
    image_kwargs=model_kwargs.pop('image_branch',{})
    xyz_kwargs=model_kwargs.pop('xyz_branch',{})
    image_branch = HighResolutionNet(MODEL_CONFIGS[arch], **image_kwargs)
    xyz_branch = HighResolutionNet(MODEL_CONFIGS[arch], **xyz_kwargs)
    model=HighResolutionNet_image_xyz(image_branch,xyz_branch,intermediate if intermediate else 'stage4')
    return model

def hrnet18_image_xyz(**kwargs):
    r"""HRNet-18 model
    """
    return _hrnet('hrnet18', **kwargs)

def hrnet32_image_xyz(**kwargs):
    r"""HRNet-32 model
    """
    return _hrnet('hrnet32', **kwargs)

def hrnet48_image_xyz(**kwargs):
    r"""HRNet-48 model
    """
    return _hrnet('hrnet48', **kwargs)

# def mvmapper(**kwargs):
#     return _hrnet('mvmapper', **kwargs)

if __name__=='__main__':
    from utils.freeze_model_func import *
    model_kwargs={'image_branch':{'in_channels':2,'out_channels':1},'xyz_branch':{'in_channels':1,'out_channels':1}}
    model=hrnet32_image_xyz(**model_kwargs)
    # image_x=torch.rand([10,3,512,512])
    # xyz_x=torch.rand([4,3,512,512])
    # batch_info=[3,2,3,2]
    # grid=torch.rand([10,512,512,2])
    # # y=model([image_x,xyz_x])
    # y=model({'image':[image_x,xyz_x],'grid':grid,'batch_info':batch_info})
    freeze_by_names(model,['image_branch'])
    # freeze_bn_model(model,'image_branch')
    # for name,param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # print("ok")
    print(len(list(model.parameters())))
