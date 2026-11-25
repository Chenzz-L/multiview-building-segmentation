# -*- coding: utf-8 -*-
import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch.utils.data as tdata

from data.transforms.augmentation import seg_augmentation,build_agument_pipeline
from data.datasets.dataste_support import *
from data.datasets.dataset_Base import *
# from utils.raster_io import RasterIO
from utils.make_samples import *

__all__ = ['Dataset_RGB_selectviews']

@regist_dataset()
class Dataset_RGB_selectviews(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 augmentation: bool = True,
                 standardization: bool = True,
                 mean: list = None,
                 std: list = None,
                 clip: list = None,
                 data_pipeline=None,
                 indices_name= None,
                 max_views: int = 25):
        r"""
        :param data_dir: root dir of dataset, including image, label inside
        :param augmentation: if need data augmentation
        :param clip: clip indices if needed
        usage:
            train_set = SegmentationDataset(data_dir=trainset_dir)
            # directly set statics from outside
            train_set.set_stat(rgb_mean, rgb_std)
            # read stat if statics not given
                if os.path.exists(txt_file):
                    train_set.read_stat(txt_file)
            # calculate stat
            train_set.calculate_stat()
            # balance_samples
            self.add_positive_samples(positive_ratio_thresh=0.2, balance_rate=0.7, str_pr='positive_ratio')
        """
        super().__init__(data_dir, augmentation, standardization, mean,std,clip,data_pipeline,indices_name)
        self.views_meta = self.build_pair()
        self.max_views = max_views
        self.active_ids = []
        self.resample_views()
        
    def build_pair(self):
        metas = make_pair_with_indices(self.ids)
        return metas
    
    def resample_views(self, seed=None):
        """
        每个Epoch调用一次
        遍历所有 Patch Group，随机选取 max_views 个，然后扁平化放入 active_ids
        """
        self.active_ids = []  # 清空旧列表
        if seed is not None:
            random.seed(seed)
        for group in self.views_meta:
            # group 是一个包含文件名的列表
            if len(group) > self.max_views:
                selected = random.sample(group, self.max_views)
            else:
                selected = group[:]  # 不足则全选
            
            # 扁平化添加到当前列表
            self.active_ids.extend(selected)
        
        if seed is not None:
            random.shuffle(self.active_ids)
            
        print(f"🔄 Resampled: Current epoch contains {len(self.active_ids)} images.")
    def __len__(self):
        # 覆盖父类的 len，返回当前采样后的长度
        return len(self.active_ids)
    
    def __getitem__(self, idx):
        if self.standardization:
            mean = self.mean
            std = self.std
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)

        current_filename = self.active_ids[idx]
        # read image
        # image_data=self.open(os.path.join(self._image_dir, self.ids[idx]),
        #                     channel_sum=3,
        #                     standardization=self.standardization,mean=0,std=255.0)
        
        # image_data=self.open(os.path.join(self._image_dir, self.ids[idx]),
        #                      channel_sum=3,standardization=self.standardization,mean=mean,std=std)
        image_data=self.open(os.path.join(self._image_dir, current_filename),
                             channel_sum=3)
        # read label
        label_data=self.open(os.path.join(self._label_dir, current_filename))
            
        # data augmentation
        # if self.augmentation:
            # image_data, label_data = seg_augmentation(aug=self.data_pipeline,
            #                                           image=image_data,
            #                                           label=label_data)
        
        if self.augmentation and self.data_pipeline!=None:
            image_data, label_data = seg_augmentation(image=image_data,
                                                      mask=label_data,
                                                      aug=self.data_pipeline)
            
        return image_data, label_data

# if __name__=='__main__':
#     from utils.raster_io import fastly_save_image
#     import gdal
#     aug=[{'type':'ToFloat','max_value':125.0},
#          {'type':'ColorJitter','always_apply':True}]
#     dataset=Dataset_RGB(data_dir=r'/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/test/samples_colinear',
#                         augmentation=True,
#                         standardization=True,
#                         data_pipeline=aug)
#     data_loader = tdata.DataLoader(dataset=dataset, batch_size=1,
#                                    num_workers=0, shuffle=True)
#     batch_iterator=iter(data_loader)
#     for batch in batch_iterator:
#         img,label=batch
#         fastly_save_image(img.squeeze(0).numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/image.tif',gdal_type=gdal.GDT_Float32)
#         fastly_save_image(label.squeeze(0).numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/label.tif',gdal_type=gdal.GDT_Float32)
#         break
