import os
import sys
import random
import numpy as np
import pandas as pd
import torch

from data.transforms.augmentation import seg_augmentation_imageonly, build_agument_pipeline,seg_augmentation
from data.datasets.data_utils import expand_data_2_BCHW,expand_datalist_2_BCHW
from data.datasets.dataste_support import *
from data.datasets.dataset_Base import *
from utils.make_samples import *

__all__ = ['Inference_Dataset_DSM_Fusion_RGB_DSM_feature']

@regist_dataset()
class Inference_Dataset_DSM_Fusion_RGB_DSM_feature(BaseDataset):
    def __init__(self,
                 augmentation: bool,
                 feat_data_dir: str,
                 xyz_data_dir: str,
                 indices_name='indices',
                 standardization=True,
                 xyz_dsm_none_value=-99999,
                 clip=None,
                 data_pipeline=None):

        super().__init__(xyz_data_dir, augmentation, standardization,
                         clip, data_pipeline, indices_name)

        self.feat_data_dir = feat_data_dir
        self.xyz_data_dir  = xyz_data_dir

        self._image_dir_name = 'image'
        self._label_dir_name = 'label'
        self._dsm_dir_name   = 'dsm'

        # self.indices = pd.read_csv(
        #     os.path.join(xyz_data_dir, indices_name + '.csv')
        # )
        # self.ids = self.indices['name'].tolist()

        if clip is not None:
            self.ids = self.ids[clip[0]:clip[1]]

        self.xyz_dsm_none_value = xyz_dsm_none_value

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = os.path.splitext(self.ids[idx])[0]
        label_path = os.path.join(self.xyz_data_dir, self._label_dir_name, name +'.tif')
        img_path   = os.path.join(self.xyz_data_dir, self._image_dir_name, name +'.tif')
        dsm_path   = os.path.join(self.xyz_data_dir, self._dsm_dir_name,   name+'.tif')

        xyz_label = self.open(label_path)
        xyz_image = self.open(img_path, channel_sum=3)
        xyz_dsm   = self.open(dsm_path, none_value=self.xyz_dsm_none_value)

        # 恢复被误删的数据增强逻辑
        if self.augmentation and self.data_pipeline is not None:
            xyz_image, xyz_label, xyz_dsm = seg_augmentation(
                image=xyz_image,
                mask=[xyz_label, xyz_dsm],
                aug=self.data_pipeline
            )

        # ---------- feature (适配 k1，保持 3D 输出和 5 个返回值) ----------
        feat_path = os.path.join(self.feat_data_dir, name + '.pt')
        KEEP_KEYS = ("res2", "res3", "res4", "res5")

        if os.path.exists(feat_path):
            feat_raw = torch.load(feat_path)
            
            # 兼容 k1 的 clusters 格式
            feat_source = feat_raw['clusters'] if 'clusters' in feat_raw else feat_raw

            feat_fused = {}
            for k in KEEP_KEYS:
                if k in feat_source:
                    f = feat_source[k]
                    # 如果是 [1, C, H, W]，压回 [C, H, W] 以适配原版格式
                    if f.ndim == 4 and f.shape[0] == 1:
                        f = f.squeeze(0)
                    feat_fused[k] = f
        else:
            feat_fused = {
                "res2": torch.zeros((256, 128, 128), dtype=torch.float32),
                "res3": torch.zeros((256, 64, 64), dtype=torch.float32),
                "res4": torch.zeros((256, 32, 32), dtype=torch.float32),
                "res5": torch.zeros((256, 16, 16), dtype=torch.float32)
            }

        # ---------- to tensor ----------
        xyz_image = torch.from_numpy(xyz_image).float()
        xyz_dsm   = torch.from_numpy(xyz_dsm).float()
        xyz_label = torch.from_numpy(xyz_label).long()

        return feat_fused, xyz_image, xyz_dsm, xyz_label, name
