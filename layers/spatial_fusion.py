from typing import List,Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
# os.environ['CUDA_VISIBLE_DEVICES']='5'


class SpatialFusion1D(nn.Module):
    def __init__(self,img_height,img_width):
        """spatialfusion with the N*HW*C input

        Args:
            img_height (int): the input img height
            img_width (int): the input img width
        """
        super(SpatialFusion1D, self).__init__()
        self.img_height=img_height
        self.img_width=img_width
    def forward(self,x,grid,batch_info):
        assert x.shape[-2]==self.img_height*self.img_width,"input shape error!!!"
        x_channel=x.shape[-1]
        x=x.permute(0,2,1)
        x=x.view(x.shape[0],-1,self.img_height,self.img_width)
        x=F.grid_sample(x,grid,mode='nearest',align_corners=True)
        x2=torch.mean(x[0:batch_info[0]],dim=0,keepdim=True)
        for i, batch in enumerate(batch_info):
            if i > 0:
                x3 = torch.mean(x[sum(batch_info[0:i]):sum(batch_info[0:i]) + batch_info[i]], dim=0, keepdim=True)
                x2 = torch.cat([x2, x3], dim=0)
        x=x2.view(batch_info.shape[0],-1,x_channel)
        return x

class SpatialFusion2D(nn.Module):
    def __init__(self,in_channels,mode='nearest',align_corners=True):
        super(SpatialFusion2D, self).__init__()
        self.in_channels=in_channels
        self.mode=mode
        self.align_corners=align_corners
        self._init_input()

    def _init_input(self):
        if isinstance(self.in_channels,(List,Tuple)):
            self.resize_grid=True
        else:
            self.resize_grid=False

    def group_to_single_view(self,x,batch_info):
        # if not self.resize_grid:
        #     x=(x,)
        # x_out=[]
        # for xi in x:
        #     xi2=torch.mean(xi[0:batch_info[0]],dim=0,keepdim=True)
        #     for i, batch in enumerate(batch_info):
        #         if i > 0:
        #             xi3 = torch.mean(xi[sum(batch_info[0:i]):sum(batch_info[0:i]) + batch_info[i]], dim=0, keepdim=True)
        #             xi2 = torch.cat([xi2, xi3], dim=0)
        #     x_out.append(xi2)

        # if self.resize_grid:
        #     return x_out
        # else:
        #     return x_out[0]

        if not self.resize_grid:
            x=(x,)
        x_out=[]
        for xi in x:
            x_splits=xi.split(batch_info,dim=0)
            x_split=[torch.mean(x_split,dim=0,keepdim=True) for x_split in x_splits]
            x_split=torch.concat(x_split,dim=0)
            x_out.append(x_split)

        if self.resize_grid:
            return x_out
        else:
            return x_out[0]

    def forward(self,x,grid,batch_info):
        grid_shape=grid.shape[1:3]
        if self.resize_grid:
            for i in range(len(x)):
                if grid_shape!=x[i].shape[2:]:
                    gridi=grid.transpose(1,3)
                    gridi=F.interpolate(gridi,
                                        size=x[i].shape[2:],
                                        mode='bilinear',
                                        align_corners=self.align_corners)
                    gridi=gridi.transpose(1,3)
                else:
                    gridi=grid
                x[i]=F.grid_sample(x[i],gridi,mode=self.mode,align_corners=self.align_corners)

        else:
            x=F.grid_sample(x,grid,mode=self.mode,align_corners=self.align_corners)

        x=self.group_to_single_view(x,batch_info)
        return x

class SpatialFusion2DV2(nn.Module):
    def __init__(self,in_channels,mode='nearest',align_corners=True,resize_feature=False):
        """像方到物方的转换融合
        Args:
            resize_feature (bool, optional): 是否先进行采样，后缩放到特征尺寸
        """
        super(SpatialFusion2DV2, self).__init__()
        self.in_channels=in_channels
        self.mode=mode
        self.align_corners=align_corners
        self.resize_feature=resize_feature
        self._init_input()

    def _init_input(self):
        if isinstance(self.in_channels,(List,Tuple)):
            self.multi_input=True
        else:
            self.multi_input=False

    def group_to_single_view(self,x,batch_info):
        x_out=[]
        for xi in x:
            x_splits=xi.split(batch_info,dim=0)
            x_split=[torch.mean(x_split,dim=0,keepdim=True) for x_split in x_splits]
            x_split=torch.concat(x_split,dim=0)
            x_out.append(x_split)

        if self.multi_input:
            return x_out
        else:
            return x_out[0]
    
    # def group_to_single_view(self, x, batch_info):
    #     x_out = []
    #     for xi in x:
    #         # 拆分为若干组，每组形状: (view_num, C, H, W)
    #         x_splits = xi.split(batch_info, dim=0)

    #         x_split = []
    #         for x_split_i in x_splits:
    #             # mask: 1 表示有效值（非0），0 表示无效值
    #             mask = (x_split_i != 0).to(x_split_i.dtype)

    #             # 按视角维度求加权和与有效计数
    #             sum_valid = torch.sum(x_split_i * mask, dim=0, keepdim=True)   # (1, C, H, W)
    #             count_valid = torch.sum(mask, dim=0, keepdim=True)             # (1, C, H, W)

    #             # 避免除0：若该位置全部无效（count=0），则输出0
    #             fused = torch.where(count_valid > 0,
    #                                 sum_valid / count_valid,
    #                                 torch.zeros_like(sum_valid))

    #             x_split.append(fused)

    #         # 拼接所有组 → (num_groups, C, H, W)
    #         x_split = torch.concat(x_split, dim=0)
    #         x_out.append(x_split)

    #     if self.multi_input:
    #         return x_out
    #     else:
    #         return x_out[0]

    def forward(self,x,grid,batch_info):
        grid_shape=grid.shape[1:3]
        if not self.multi_input:
            x=[x]

        for i in range(len(x)):
            x_shape=x[i].shape[2:]
            if grid_shape!=x_shape:
                if self.resize_feature:
                                x[i]=F.interpolate(F.grid_sample(x[i],grid,mode=self.mode,align_corners=self.align_corners),
                                       size=x_shape,
                                       mode='bilinear',
                                       align_corners=self.align_corners)
                else:
                    # gridi=grid.transpose(1,3)
                    gridi=F.interpolate(grid.transpose(1,3),
                                        size=x_shape,
                                        mode='bilinear',
                                        align_corners=self.align_corners).transpose(1,3)
                    # gridi=gridi.transpose(1,3)
                    x[i]=F.grid_sample(x[i],gridi,mode=self.mode,align_corners=self.align_corners)
            else:
                x[i]=F.grid_sample(x[i],grid,mode=self.mode,align_corners=self.align_corners)

        x = self.group_to_single_view(x,batch_info)
        return x

class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()
    def forward(self,xyz_features, proj_features):
        assert type(xyz_features)==type(proj_features)
        if not isinstance(xyz_features, list):
           return [x + y for x, y in zip(xyz_features, proj_features)]
        return xyz_features + proj_features


# if __name__=='__main__':
#     x1=torch.rand(10,3,512,512)
#     x2=torch.rand(10,3,256,256)
#     x3=torch.rand(10,3,128,128)
#     x=[x1,x2,x3]
#     grid=torch.rand(10,512,512,2)
#     batch_info=[3,3,4]
#     spfusion=SpatialFusion2DV2((512,256,128),resize_feature=True)
#     y=spfusion(x,grid,batch_info)
#     print(y.shape)
