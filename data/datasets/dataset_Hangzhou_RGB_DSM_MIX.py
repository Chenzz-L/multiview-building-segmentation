# -*- coding: utf-8 -*-
import os
import sys
from typing import Optional,List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import pandas as pd
from torch.utils import data as tdata

from data.datasets.dataset_Base import BaseDataset
from data.datasets.dataste_support import *
from data.transforms.augmentation import train_data_augmentation_dsm,seg_augmentation,build_agument_pipeline
# from utils.raster_io import RasterIO


__all__ = ['Dataset_Hangzhou_RGB_DSM_MIX']

@regist_dataset()
class Dataset_Hangzhou_RGB_DSM_MIX(BaseDataset):
    def __init__(self,
                 data_dir: List[str],
                 augmentation: bool = True,
                 dsm_none_value: float = -99999,
                 img_none_value: float = 0,
                 mask = False,
                 clip: list = None,
                 data_pipeline = None,
                 indices_name : Optional[List[str]] = None):
        self.data_dir = data_dir
        self._image_dir = [os.path.join(x, 'image') for x in  data_dir ]
        self._dsm_dir = [os.path.join(x, 'dsm') for x in  data_dir ]
        self._label_dir = [os.path.join(x, 'label') for x in  data_dir ]
        self.dsm_none_value = dsm_none_value
        self.img_none_value = img_none_value
        self.mask = mask

        # self.data_dirs={'image':'image','label':'label'}
        self.augmentation = augmentation
        # get indices, return pd.DataFrame, a python dictionary
        if indices_name != None:
            self.indices = [pd.read_csv(os.path.join(x, f'{name}.csv')) for x,name in zip(data_dir,indices_name)]
        else:
            self.indices = [pd.read_csv(os.path.join(x, 'indices.csv')) for x in  data_dir]


        self.ids = [x['name'].tolist() for x in self.indices]

        self.ids = [os.path.join(x,name)  for x,ids in zip(data_dir,self.ids) for name in ids]
        # clip
        if clip is not None:
            assert len(clip) == 2
            self.ids = self.ids[clip[0]:clip[1]]
            
        self.data_pipeline = None
        if augmentation and data_pipeline:
            if not 'ToFloat' in data_pipeline[0].values():# 自定义ToFloat
                data_pipeline.insert(0,{'type':'ToFloat','max_value':1})
            self.data_pipeline = build_agument_pipeline(data_pipeline)
        else:
            self.augmentation = False

    def __getitem__(self, idx):

        name = os.path.basename(self.ids[idx])
        current_dir=os.path.dirname(self.ids[idx])
        
        image_data=self.open(os.path.join(current_dir,'image',name),channel_sum=3,none_value=self.img_none_value)
        # read label
        label_data=self.open(os.path.join(current_dir,'label',name))
        # read dsm
        dsm_data=self.open(os.path.join(current_dir,'dsm',name),channel_sum=1,none_value=self.dsm_none_value)

        # data augmentation
        if self.augmentation and self.data_pipeline!=None:
            image_data,label_data,dsm_data=seg_augmentation(image = image_data,mask = [label_data,dsm_data],aug = self.data_pipeline)

        image_data = np.concatenate((image_data, dsm_data), axis=0)
        return image_data, label_data


if __name__=='__main__':
    from utils.raster_io import fastly_save_image
    from osgeo import gdal

    data_pipeline=[{'type' : 'RandomResizedCrop','height': 384,'width' : 384,'scale' : [0.08,1.0],
                    'ratio' : [0.75,1.3333333333333333],'interpolation' : 2,'always_apply':True},
                   {'type': 'Flip' , 'p' : 0.5},
                   {'type' : 'RandomBrightnessContrast' , 'p' : 0.5},
                   {'type' : 'ColorJitter'},
                   {'type' : 'ElasticTransform'},
                   {'type' : 'GridDistortion'}]
    dataset = Dataset_Melbourne_PAN_DSM_MIX([
        r"/path/to/nas/mvmapper_datamake/0_DeepLearning_v3/IMG_MV/XYZ_PANSharpen_samples_512_512_360_IMGObject",
        r'/path/to/nas/mvmapper_datamake/0_DeepLearning_v3/IMG_MV/XYZ_PANSharpen_samples_512_512_408_SlideWindow'],
                          augmentation = True,mask = False,clip = None,data_pipeline = data_pipeline)

    data_loader = tdata.DataLoader(dataset = dataset, batch_size=1,
                                   num_workers = 0, shuffle = False,  # cfg.shuffle,
                                   pin_memory = True)

    batch_iterator = iter(data_loader)
    for batch in batch_iterator:
        x, y= batch
    #     fastly_save_image(x.squeeze(0).numpy(),'/model/zhangpl/mvmaper/mvmapper_melbourne/image.tif',gdal_type=gdal.GDT_Float32)
    #     fastly_save_image(y.squeeze(0).numpy(),'/model/zhangpl/mvmaper/mvmapper_melbourne/label.tif',gdal_type=gdal.GDT_Float32)
    #     break
        # if torch.isnan(x).any():
        #     print('found image nan')
        # if torch.isinf(x).any():
        #     print('found image inf')
            
        # if torch.isnan(y).any():
        #     print('found label nan')
        # if torch.isinf(y).any():
        #     print('found label inf')
        # for i in range(x.shape[0]):
        #     out = x[i].numpy()
        #     fastly_save_image(out, os.path.join(r"/path/to/user/NAS/gwx/experiment_result/", 'img'+str(i) + ".tif"),
        #                       gdal_type=gdal.GDT_Float32)
        #     fastly_save_image(y[i].numpy(),
        #                       os.path.join(r"/path/to/user/NAS/gwx/experiment_result/", 'label'+str(i) + ".tif"),
        #                       gdal_type=gdal.GDT_Float32)

        # print("ok")

        # break
