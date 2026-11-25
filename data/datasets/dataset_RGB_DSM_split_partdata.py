# -*- coding: utf-8 -*-
import os
import sys
from typing import Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
from torch.utils import data as tdata

from data.datasets.dataset_Base import BaseDataset
from data.datasets.dataste_support import *
from data.transforms.augmentation import train_data_augmentation_dsm,seg_augmentation,build_agument_pipeline
# from utils.raster_io import RasterIO


__all__ = ['dataset_RGB_DSM_split_partdata']

@regist_dataset()
class dataset_RGB_DSM_split_partdata(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 augmentation: bool = True,
                 standardization: bool = True,
                 dsm_none_value: float = -99999,
                 mask=False,
                 mean: list = None,
                 std: list = None,
                 clip: list = None,
                 data_pipeline=None,
                 indices_name=None,
                 split_k: int = 5):
        
        self._dsm_dir = os.path.join(data_dir, 'dsm')
        self.dsm_none_value = dsm_none_value
        self.mask = mask
        super().__init__(data_dir, augmentation, standardization, mean, std, clip, data_pipeline, indices_name)
        self.split_k = split_k
        self.all_ids = list(self.ids)
        if self.split_k > 1:
            total_len = len(self.all_ids)
            self.fixed_length = total_len // self.split_k
            self.resample_views(seed=0)
        else:
            self.fixed_length = len(self.all_ids)

    def set_dir(self, image_dir_name: str, label_dir_name: str, dsm_dir_name: str = None):
        self._image_dir = os.path.join(self.data_dir, image_dir_name)
        self._label_dir = os.path.join(self.data_dir, label_dir_name)
        self._dsm_dir = os.path.join(self.data_dir, dsm_dir_name)
        assert os.path.isdir(self.data_dir)
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)
        assert os.path.isdir(self._dsm_dir)

    def resample_views(self, seed: int):
        """
        根据 epoch (seed) 动态切换数据集的子集。
        这里不使用随机种子，而是使用数学取模来实现确定性的轮转。
        
        Args:
            seed (int): 对应 trainer.current_epoch
        """
        if self.split_k <= 1:
            return

        offset = seed % self.split_k
        
        # 1. 先进行步长切片
        subset = self.all_ids[offset::self.split_k]
        if len(subset) > self.fixed_length:
            subset = subset[:self.fixed_length]
            
        self.ids = subset

    def __getitem__(self, idx):
        # idx 范围自动变为 0 到 len(当前子集)-1
        # self.ids 已经是被切片过的列表
        img_id = self.ids[idx]
        
        image_data = self.open(os.path.join(self._image_dir, '%s' % img_id), channel_sum=3)
        label_data = self.open(os.path.join(self._label_dir, '%s' % img_id))
        dsm_data = self.open(os.path.join(self._dsm_dir, '%s' % img_id), channel_sum=1, none_value=self.dsm_none_value)

        if self.augmentation and self.data_pipeline is not None:
            image_data, label_data, dsm_data = seg_augmentation(image=image_data, mask=[label_data, dsm_data], aug=self.data_pipeline)

        return image_data, dsm_data, label_data
    
    def __len__(self):
        # 返回当前子集的长度
        return len(self.ids)
