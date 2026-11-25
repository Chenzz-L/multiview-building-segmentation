# -*- coding: utf-8 -*-
import glob
import os
from typing import Tuple,Optional,Union

import numpy as np
import torch.utils.data as tdata

from utils.raster_io import RasterIO
from data.reading_data_tool import image_open
# from data.datasets.dataste_support import *
__all__ = ['InferenceBaseDataset']


class InferenceBaseDataset(tdata.Dataset):
    """
        与训练dataset不同的是,label_data不需要添加一个维度
    """
    def __init__(self,
                 data_dir: str,
                 standardization: bool = True,
                 mean: list = None,
                 std: list = None,
                 img_type:str='tif',
                 indices_name=None):

        assert os.path.exists(data_dir)
        self.data_dir = data_dir
        self._image_dir = os.path.join(data_dir, 'image')
        self._label_dir =os.path.join(data_dir, 'label') if os.path.exists(os.path.join(data_dir, 'label')) else None
        self.standardization=standardization
        self.img_type=img_type
        if self.standardization:
            self.mean=mean
            self.std=std
        #当标准化文件不存在时，计算该参数
        #若存在则从中读取该参数
        
        # self.stat_file = self.data_dir + '/' + 'stat_binary.txt'
        # self.stat_data=self.read_stat(self.stat_file)
        # self.ids = [os.path.basename(img) for img in glob.glob(os.path.join(data_dir,self._image_dir,"*."+img_type))]
        if indices_name != None:
            with open(os.path.join(data_dir,indices_name+'.txt'),'r') as f:
                outs=f.readlines()
                self.ids=[out.rstrip() for out in outs]
        else:
            self.ids = [os.path.basename(img) for img in glob.glob(os.path.join(data_dir,self._image_dir,"*."+img_type))]
    def open(self,data_path,channel_sum=None,scale:Optional[float]=None,
            standardization=False,mean:Optional[Union[Tuple[float],float]]=None,
            std:Optional[Union[Tuple[float],float]]=None,
            mask=False,
            mask_value:Optional[float]=None,
            none_value:Optional[float]=None,
            xoff:Optional[int]=None,yoff:Optional[int]=None,
            w:Optional[int]=None,h:Optional[int]=None):   
        return image_open(data_path,channel_sum,scale,
                          standardization,mean,std,
                          mask,mask_value,none_value,
                          xoff,yoff,w,h)
        
    def set_dir(self, image_dir_name: str, label_dir_name: str):
        self._image_dir = os.path.join(self.data_dir, image_dir_name)
        self._label_dir = os.path.join(self.data_dir, label_dir_name)
        assert os.path.isdir(self.data_dir)
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)
        self.ids = [os.path.basename(img) for img in
                    glob.glob(os.path.join(self.data_dir, self._image_dir, "*." + self.img_type))]

    def set_stat(self, stat_data):
        r"""
        :param rgb_mean: Tuple[Any,Any,Any]  # rgb_mean = [R_mean, G_mean, B_mean]
        :param rgb_std: Tuple[Any,Any,Any], # rgb_std = [R_std, G_std, B_std]
        """
        self.stat_data=stat_data

    def read_stat(self, stat_file):
        assert os.path.isfile(stat_file)
        def clear_fix(strs):
            return [float(s.strip()) for s in strs]
        with open(stat_file, "r") as f:
            lines = f.readlines()
            stat_data=tuple([clear_fix(line.split(',')) for line in lines])
        return stat_data

    def __len__(self):
        return len(self.ids)
