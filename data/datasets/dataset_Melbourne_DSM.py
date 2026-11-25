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


__all__ = ['Dataset_Melbourne_DSM']

@regist_dataset()
class Dataset_Melbourne_DSM(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 augmentation: bool = True,
                 dsm_none_value: float = -32767,
                 mask=False,
                 clip: list = None,
                 data_pipeline=None,
                 indices_name=None):
        self._dsm_dir = os.path.join(data_dir, 'dsm')
        self.dsm_none_value = dsm_none_value
        self.mask = mask
        super().__init__(data_dir,augmentation,False,clip,data_pipeline,indices_name)

        self.data_pipeline=None
        if augmentation and data_pipeline:
            if not 'ToFloat' in data_pipeline[0].values():# 自定义ToFloat
                data_pipeline.insert(0,{'type':'ToFloat','max_value':1})
            self.data_pipeline=build_agument_pipeline(data_pipeline)
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

        image_data=self.open(os.path.join(self._image_dir, '%s' % self.ids[idx]),channel_sum=1)
        # read label
        label_data=self.open(os.path.join(self._label_dir, '%s' % self.ids[idx])).astype(np.int64)
        # read dsm
        dsm_data=self.open(os.path.join(self._dsm_dir, '%s' % self.ids[idx]),channel_sum=1)

        # data augmentation
        if self.augmentation and self.data_pipeline!=None:
            image_data,label_data,dsm_data = seg_augmentation(image=image_data, mask=[label_data,dsm_data], aug = self.data_pipeline)

        image_data = np.concatenate((image_data, dsm_data), axis=0)
        return image_data, label_data


# if __name__=='__main__':
#     from utils.raster_io import fastly_save_image
#     import gdal

#     data_pipeline=[{'type' : 'RandomResizedCrop','height': 384,'width' : 384,'scale' : [0.08,1.0],
#                     'ratio' : [0.75,1.3333333333333333],'interpolation' : 2,'always_apply':True},
#                    {'type': 'Flip' , 'p' : 0.5},
#                    {'type' : 'RandomBrightnessContrast' , 'p' : 0.5},
#                    {'type' : 'ColorJitter'},
#                    {'type' : 'ElasticTransform'},
#                    {'type' : 'GridDistortion'}]
#     dataset = Dataset_Melbourne_DSM(r"/path/to/user/NAS/gwx/DeepL/Melbourne/IMG_MV/samples",
#                           False,-99999,False,None,data_pipeline,indices_name='train_indices')

#     data_loader = tdata.DataLoader(dataset=dataset, batch_size=1,
#                                    num_workers=0, shuffle=False,  # cfg.shuffle,
#                                    pin_memory=True)

#     batch_iterator = iter(data_loader)
#     for batch in batch_iterator:
#         x, y= batch
#         fastly_save_image(x.squeeze(0).numpy(),'/path/to/user/NAS/gwx/DeepL/Melbourne/image.tif',gdal_type=gdal.GDT_Float32)
#         fastly_save_image(y.squeeze(0).numpy(),'/path/to/user/NAS/gwx/DeepL/Melbourne/label.tif',gdal_type=gdal.GDT_Float32)
#         break
#         if torch.isnan(x).any():
#             print('found image nan')
#         if torch.isinf(x).any():
#             print('found image inf')
            
#         if torch.isnan(y).any():
#             print('found label nan')
#         if torch.isinf(y).any():
#             print('found label inf')
#         for i in range(x.shape[0]):
#             out = x[i].numpy()
#             fastly_save_image(out, os.path.join(r"/path/to/user/NAS/gwx/experiment_result/", 'img'+str(i) + ".tif"),
#                               gdal_type=gdal.GDT_Float32)
#             fastly_save_image(y[i].numpy(),
#                               os.path.join(r"/path/to/user/NAS/gwx/experiment_result/", 'label'+str(i) + ".tif"),
#                               gdal_type=gdal.GDT_Float32)

#         print("ok")

#         break
