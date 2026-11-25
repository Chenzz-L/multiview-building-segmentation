import os
import random
import torch
import numpy as np
import pandas as pd

from data.transforms.augmentation import seg_augmentation
from data.datasets.dataste_support import regist_dataset
from data.datasets.dataset_Base import BaseDataset
from utils.make_samples import make_pair_with_xyz_indices, read_xyz_img_pair_file

@regist_dataset()
class Dataset_DSM_Fusion_RGB_DSM_RandomSelect_CrossAttn(BaseDataset):
    """
    随机视角采样 Dataset。
    训练时从原始的所有视角特征中随机抽取 max_k 个，不进行聚类平均，
    强制模型在多样性极高的原始特征中学会“选优”。
    """
    def __init__(self,
                 augmentation: bool,
                 feat_data_dir: str,
                 xyz_data_dir: str,
                 xyz_img_pair_file: str,
                 indices_name='indices',
                 standardization=True,
                 xyz_dsm_none_value=-99999,
                 clip=None,
                 data_pipeline=None,
                 max_k=6,
                 xyz_num_length=3,
                 random_select=True):

        super().__init__(xyz_data_dir, augmentation, standardization,
                         clip, data_pipeline, indices_name)

        self.feat_data_dir = feat_data_dir
        self.xyz_data_dir  = xyz_data_dir
        self.max_k = max_k
        self.xyz_num_length = xyz_num_length
        self.random_select = random_select

        self._image_dir_name = 'image'
        self._label_dir_name = 'label'
        self._dsm_dir_name   = 'dsm'
        self.xyz_dsm_none_value = xyz_dsm_none_value

        # 1. 建立 TDOM -> 原始多视角特征的映射
        # 这里使用 make_pair_with_xyz_indices，它会扫描 feat_data_dir 下所有的原始 .pt 文件
        self.pairs_meta = make_pair_with_xyz_indices(
            xyz_files=self.ids,
            feat_dir=self.feat_data_dir,
            xyz_num_length=self.xyz_num_length
        )
        
        # 2. 读取 XYZ 到具体文件名的映射（处理不同区域的 key）
        self.xyz_img_dict = read_xyz_img_pair_file(xyz_img_pair_file)

    def __len__(self):
        return len(self.pairs_meta)

    def __getitem__(self, idx):
        xyz_file, all_view_files = self.pairs_meta[idx]
        
        # ---------- 1. 加载 目标域 (TDOM) 的基础数据 ----------
        xyz_name = os.path.splitext(os.path.basename(xyz_file))[0]
        # 解析杭州特有的文件名规则
        parts = xyz_name.split("_")
        xyz_num = parts[-3].split("-")[-1]
        num_suffix = "_".join(parts[-2:])
        xyz_tif_name = f"{self.xyz_img_dict[xyz_num]}_{num_suffix}.tif"

        label_path = os.path.join(self.xyz_data_dir, self._label_dir_name, xyz_tif_name)
        img_path   = os.path.join(self.xyz_data_dir, self._image_dir_name, xyz_tif_name)
        dsm_path   = os.path.join(self.xyz_data_dir, self._dsm_dir_name,   xyz_tif_name)

        xyz_label = self.open(label_path)
        xyz_image = self.open(img_path, channel_sum=3)
        xyz_dsm   = self.open(dsm_path, none_value=self.xyz_dsm_none_value)

        if self.augmentation and self.data_pipeline is not None:
            xyz_image, xyz_label, xyz_dsm = seg_augmentation(
                image=xyz_image,
                mask=[xyz_label, xyz_dsm],
                aug=self.data_pipeline
            )

        xyz_image = torch.from_numpy(xyz_image).float()
        xyz_dsm   = torch.from_numpy(xyz_dsm).float()
        xyz_label = torch.from_numpy(xyz_label).long()

        # ---------- 2. 随机采样 max_k 个视角 ----------
        if self.random_select and len(all_view_files) > self.max_k:
            selected_views = random.sample(all_view_files, self.max_k)
        else:
            selected_views = all_view_files[: self.max_k]
        
        # ---------- 3. 加载特征并封装为 List[Dict] ----------
        all_cam_features = []
        all_cam_exist = []
        KEEP_KEYS = ("res2", "res3", "res4", "res5")
        SHAPES = {"res2":(256,128,128), "res3":(256,64,64), "res4":(256,32,32), "res5":(256,16,16)}

        for i in range(self.max_k):
            view_dict = {}
            is_valid = False
            
            if i < len(selected_views):
                feat_path = os.path.join(self.feat_data_dir, selected_views[i])
                try:
                    data = torch.load(feat_path, map_location='cpu')
                    for scale in KEEP_KEYS:
                        view_dict[scale] = data[scale]
                    is_valid = True
                except Exception:
                    is_valid = False
            
            # 如果无效或超出范围，补零
            if not is_valid:
                for scale in KEEP_KEYS:
                    c, h, w = SHAPES[scale]
                    view_dict[scale] = torch.zeros((c, h, w), dtype=torch.float32)
            
            all_cam_features.append(view_dict)
            all_cam_exist.append(is_valid)

        all_cam_exist = torch.tensor(all_cam_exist, dtype=torch.bool)

        # 返回格式：特征列表, 掩码, RGB, DSM, Label, 文件名
        return all_cam_features, all_cam_exist, xyz_image, xyz_dsm, xyz_label, xyz_name
