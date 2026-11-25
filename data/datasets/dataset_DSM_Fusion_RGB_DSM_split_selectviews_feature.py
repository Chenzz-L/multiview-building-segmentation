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

__all__ = ['Dataset_DSM_Fusion_RGB_DSM_split_selectviews_feature']

@regist_dataset()
class Dataset_DSM_Fusion_RGB_DSM_split_selectviews_feature(BaseDataset):
    def __init__(self,
                 augmentation: bool,
                 feat_data_dir: str,
                 xyz_data_dir: str,
                 xyz_img_pair_file: str,
                 standardization: bool = True,
                 xyz_dsm_none_value: float = -99999,
                 mask=False,
                 clip: list = None,
                 data_pipeline=None,
                 indices_name='indices',
                 max_views: int = 15):
        super().__init__(xyz_data_dir, augmentation, standardization,clip,data_pipeline,indices_name)
        assert os.path.exists(feat_data_dir)
        assert os.path.exists(xyz_data_dir)

        self.feat_data_dir = feat_data_dir
        self.xyz_data_dir = xyz_data_dir

        self._image_dir_name = 'image'
        self._label_dir_name = 'label'
        self._dsm_dir_name   = 'dsm'

        self.indices = pd.read_csv(os.path.join(xyz_data_dir, indices_name + '.csv'))
        self.ids = self.indices['name'].tolist()
        self.xyz_dsm_none_value = xyz_dsm_none_value
        self.mask = mask

        # === 1. patch group 构建 ===
        self.all_views_meta = self.build_pair()
        self.xyz_img_dict = self.read_xyz_img_pair_file(xyz_img_pair_file)

        if clip is not None:
            self.all_views_meta = self.all_views_meta[clip[0]:clip[1]]

        # === 2. 视角随机采样 ===
        self.max_views = max_views
        self.active_views_meta = []
        self.resample_views()

    def build_pair(self):
        return make_pair_with_xyz_indices(
            xyz_files=self.ids,
            feat_dir=self.feat_data_dir,
            xyz_num_length=3
        )

    def resample_views(self, seed=None):
        self.active_views_meta = []
        if seed is not None:
            random.seed(seed)

        for xyz_file, group in self.all_views_meta:
            if len(group) > self.max_views:
                selected = random.sample(group, self.max_views)
            else:
                selected = group[:]
            self.active_views_meta.append((xyz_file, selected))

        # print(f"🔄 Resampled {len(self.active_views_meta)} patch groups, "
        #       f"max views = {self.max_views}")

    def __len__(self):
        return len(self.active_views_meta)

    def __getitem__(self, idx):
        # ---------- 1. 当前 patch group 的视角 ----------
        xyz_file, current_views = self.active_views_meta[idx]

        # ---------- 2. 读取 xyz 数据（一次） ----------
        xyz_name = os.path.splitext(os.path.basename(xyz_file))[0]
        xyz_num = xyz_name.split("_")[-3].split("-")[-1]
        num = "_".join(xyz_name.split("_")[-2:])
        xyz_suffix = f"{self.xyz_img_dict[xyz_num]}_{num}.tif"

        label_path = os.path.join(self.xyz_data_dir, self._label_dir_name, xyz_suffix)
        img_path   = os.path.join(self.xyz_data_dir, self._image_dir_name, xyz_suffix)
        dsm_path   = os.path.join(self.xyz_data_dir, self._dsm_dir_name, xyz_suffix)

        xyz_label = self.open(label_path)          # [H, W]
        xyz_image = self.open(img_path, channel_sum=3)  # [3, H, W]
        xyz_dsm   = self.open(
            dsm_path,
            mask_value=self.xyz_dsm_none_value,
            none_value=self.xyz_dsm_none_value
        )  # [H, W]

        if self.augmentation and self.data_pipeline is not None:
            xyz_image, xyz_label, xyz_dsm = seg_augmentation(
                image=xyz_image,
                mask=[xyz_label, xyz_dsm],
                aug=self.data_pipeline
            )

        # ---------- 3. 读取 + 融合像方特征 ----------
        feats = []
        for file in current_views:
            feat_name = os.path.splitext(file)[0] + ".pt"
            feat_path = os.path.join(self.feat_data_dir, feat_name)
            feats.append(torch.load(feat_path))  # dict: {res2,res3,res4}

        # feats: List[Dict[str, Tensor[C,H,W]]]
        feat_fused = {}
        if len(feats) > 0:
            for k in feats[0].keys():
                # [N, C, H, W] -> mean -> [C, H, W]
                feat_fused[k] = torch.stack(
                    [f[k] for f in feats], dim=0
                ).mean(dim=0)
        else:
            # Generate empty features if no features are found
            feat_fused = {
                "res2": torch.zeros((256, 128, 128), dtype=torch.float32),
                "res3": torch.zeros((256, 64, 64), dtype=torch.float32),
                "res4": torch.zeros((256, 32, 32), dtype=torch.float32),
                "res5": torch.zeros((256, 16, 16), dtype=torch.float32)
            }

        # ---------- 4. 转 Tensor（不引入 batch 维） ----------
        xyz_image = torch.from_numpy(xyz_image).float()   # [3, H, W]
        xyz_dsm   = torch.from_numpy(xyz_dsm).float()  # [1, H, W]
        xyz_label = torch.from_numpy(xyz_label).long()    # [H, W] 或 [1,H,W] 看 loss

        return (
            feat_fused,   # dict[str, Tensor[C,H,W]]
            xyz_image,    # [3, H, W]
            xyz_dsm,      # [1, H, W]
            xyz_label     # [H, W] or [1, H, W]
        )
