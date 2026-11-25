import os
import sys
import random
import numpy as np
import pandas as pd
import torch

from data.transforms.augmentation import seg_augmentation_imageonly, build_agument_pipeline
from data.datasets.data_utils import expand_data_2_BCHW,expand_datalist_2_BCHW
from data.datasets.dataste_support import *
from data.datasets.dataset_Base import *
from utils.make_samples import *

__all__ = ['Dataset_DSM_Fusion_image_xyz_selectviews']

@regist_dataset()
class Dataset_DSM_Fusion_image_xyz_selectviews(BaseDataset):
    def __init__(self,
                 img_data_dir: str,
                 grid_dir: str,
                 xyz_data_dir: str,
                 xyz_img_pair_file: str,
                 grid_scale: float,
                 augmentation: bool = False,
                 standardization: bool = True,
                 img_dsm_none_value: float = 0,
                 xyz_dsm_none_value: float = 0,
                 mask=False,
                 clip: list = None,
                 data_pipeline=None,
                 indices_name='indices',
                 max_views: int = 25):  # 新增 max_views 参数
        
        assert os.path.exists(img_data_dir)
        assert os.path.exists(xyz_data_dir)
        assert os.path.exists(grid_dir)
        
        self.img_data_dir = img_data_dir
        self.xyz_data_dir = xyz_data_dir
        self.gird_dir = grid_dir
        self._image_dir_name = 'image'
        self._label_dir_name = 'label'
        self._dsm_dir_name = 'dsm'
        
        self.augmentation = augmentation
        # get indices
        self.indices = pd.read_csv(os.path.join(img_data_dir, indices_name + '.csv'))
        self.standardization = standardization
        self.ids = self.indices['name'].tolist()

        self.img_dsm_none_value = img_dsm_none_value
        self.xyz_dsm_none_value = xyz_dsm_none_value
        self.mask = mask
        
        # 1. 构建原始的全量配对信息
        self.all_views_meta = self.build_pair()
        
        self.xyz_img_dict = self.read_xyz_img_pair_file(xyz_img_pair_file)
        self.grid_scale = grid_scale
        
        if clip is not None:
            assert len(clip) == 2
            self.all_views_meta = self.all_views_meta[clip[0]:clip[1]]

        # 数据增强构建
        if augmentation and data_pipeline:
            if not 'ToFloat' in data_pipeline[0].values():
                data_pipeline.insert(0, {'type': 'ToFloat', 'max_value': 255})
            self.data_pipeline = build_agument_pipeline(data_pipeline, True)
            self.augmentation = augmentation
        else:
            self.augmentation = False

        # 2. 初始化视角选择逻辑
        self.max_views = max_views
        self.active_views_meta = [] # 用于存储当前 Epoch 选中的视角组合
        self.resample_views() # 初始化时先采样一次

    def build_pair(self):
        # 保持原有的构建逻辑
        metas = make_pair_with_indices(self.ids)
        return metas

    def resample_views(self, seed=None):
        """
        随机重采样每个 Patch Group 的视角。
        生成的 self.active_views_meta 是一个 list of lists，
        结构与原始 self.views_meta 相同，但子列表长度不超过 max_views。
        """
        self.active_views_meta = []
        if seed is not None:
            random.seed(seed)
        
        for group in self.all_views_meta:
            # group 是一个包含该 patch 下所有视角文件名的列表
            if len(group) > self.max_views:
                selected = random.sample(group, self.max_views)
            else:
                selected = group[:]  # 视角不足则全选
            
            self.active_views_meta.append(selected)
            
        print(f"🔄 Resampled: Dataset contains {len(self.active_views_meta)} groups, "
              f"each with max {self.max_views} views.")

    def __len__(self):
        # 长度依然是 Patch Group 的数量
        return len(self.active_views_meta)
    
    def __getitem__(self, idx):
        imgs, labels, grids = [], [], []

        # 获取当前 idx 对应的视角列表（这是经过 resample_views 筛选过的子集）
        current_views = self.active_views_meta[idx]

        # === 1. 读取 xyz 数据 ===
        # 注意：这里需要根据 current_views 中的第一个文件来解析 xyz 对应的文件名
        # 假设同一个 group 内的任何一个视角文件名都能解析出相同的 xyz 信息
        image_name = os.path.splitext(os.path.basename(current_views[0]))[0]
        
        xyz_num = image_name.split("_")[-3].split("-")[-1]
        num = "_".join(image_name.split("_")[-2:])
        xyz_suffix = f"{self.xyz_img_dict[xyz_num]}_{num}.tif"

        label_path = os.path.join(self.xyz_data_dir, self._label_dir_name, xyz_suffix)
        img_path   = os.path.join(self.xyz_data_dir, self._image_dir_name, xyz_suffix)
        dsm_path   = os.path.join(self.xyz_data_dir, self._dsm_dir_name, xyz_suffix)

        xyz_label_data = self.open(label_path)
        xyz_image_data = self.open(img_path, channel_sum=3)

        if self.augmentation and self.data_pipeline is not None:
            xyz_image_data = seg_augmentation_imageonly(xyz_image_data, self.data_pipeline)

        xyz_dsm_data = self.open(
            dsm_path,
            mask=self.mask,
            mask_value=self.xyz_dsm_none_value,
            none_value=self.xyz_dsm_none_value
        )
        xyz_image_data = np.concatenate((xyz_image_data, xyz_dsm_data), axis=0)

        # === 2. 读取多视角影像 (只读取选中的 current_views) ===
        for file in current_views:
            image_data = self.open(os.path.join(self.img_data_dir, self._image_dir_name, file),
                                channel_sum=3)
            if self.augmentation and self.data_pipeline is not None:
                image_data = seg_augmentation_imageonly(image_data, self.data_pipeline)

            dsm_data = self.open(os.path.join(self.img_data_dir, self._dsm_dir_name, file),
                                mask=self.mask,
                                mask_value=self.img_dsm_none_value,
                                none_value=self.img_dsm_none_value)
            image_data = np.concatenate((image_data, dsm_data), axis=0)
            imgs.append(image_data)

            label = self.open(os.path.join(self.img_data_dir, self._label_dir_name, file))
            labels.append(label.astype(np.int16))

            grid = self.open(os.path.join(self.gird_dir, file), scale=self.grid_scale)
            grids.append(grid)

        # === 3. 统一格式 ===
        expand_datalist_2_BCHW(imgs)
        expand_datalist_2_BCHW(labels)
        expand_datalist_2_BCHW(grids)

        xyz_label_data = expand_data_2_BCHW(xyz_label_data)
        xyz_image_data = expand_data_2_BCHW(xyz_image_data)

        # 拼接成大 tensor
        img = np.concatenate(imgs, axis=0)
        grid_data = np.concatenate(grids, axis=0)
        label_data = np.concatenate(labels, axis=0)

        # grid 维度保护性转置
        if grid_data.ndim == 4:
            grid_data = grid_data.transpose((0, 2, 3, 1))
        
        # === 4. 转 Tensor ===
        img_tensor = torch.from_numpy(img.copy()).float()
        xyz_img_tensor = torch.from_numpy(xyz_image_data.copy()).float()
        grid_tensor = torch.from_numpy(grid_data.copy()).float()
        label_tensor = torch.from_numpy(label_data.copy()).long()
        xyz_label_tensor = torch.from_numpy(xyz_label_data.copy()).long()

        return img_tensor, xyz_img_tensor, grid_tensor, label_tensor, xyz_label_tensor
