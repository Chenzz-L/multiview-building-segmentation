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


__all__ = ['Dataset_DSM']

@regist_dataset()
class Dataset_DSM(BaseDataset):
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
                 indices_name= None,):
        self._dsm_dir = os.path.join(data_dir, 'dsm')
        self.dsm_none_value = dsm_none_value
        self.mask = mask
        super().__init__(data_dir, augmentation, standardization, mean,std,clip,data_pipeline,indices_name)
        
        self.data_pipeline = None
        if augmentation and data_pipeline:
            if not 'ToFloat' in data_pipeline[0].values():# 自定义ToFloat
                data_pipeline.insert(0,{'type':'ToFloat','max_value':1})
            self.data_pipeline = build_agument_pipeline(data_pipeline)
        else:
            self.augmentation=False

    def set_dir(self, image_dir_name: str, label_dir_name: str, dsm_dir_name: str = None):
        self._image_dir = os.path.join(self.data_dir, image_dir_name)
        self._label_dir = os.path.join(self.data_dir, label_dir_name)
        self._dsm_dir = os.path.join(self.data_dir, dsm_dir_name)
        assert os.path.isdir(self.data_dir)
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)
        assert os.path.isdir(self._dsm_dir)

    def __getitem__(self, idx):
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)
        assert os.path.isdir(self._dsm_dir)
        
        image_data=self.open(os.path.join(self._image_dir, '%s' % self.ids[idx]),channel_sum=3,none_value=self.dsm_none_value)
        # read label
        label_data=self.open(os.path.join(self._label_dir, '%s' % self.ids[idx]))
        # read dsm
        dsm_data=self.open(os.path.join(self._dsm_dir, '%s' % self.ids[idx]),channel_sum=1,none_value=self.dsm_none_value)

        # data augmentation
        if self.augmentation and self.data_pipeline!=None:
            image_data,label_data,dsm_data=seg_augmentation(image=image_data,mask=[label_data,dsm_data],aug=self.data_pipeline)

        image_data = np.concatenate((image_data, dsm_data), axis=0)
        return image_data, label_data
