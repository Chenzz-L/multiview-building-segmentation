"""
Inference Dataset: 加载 5 个相机的 per-camera 融合特征
返回额外的 name 字段用于推理解析 offset
"""
import os
import torch
import numpy as np

from data.transforms.augmentation import seg_augmentation
from data.datasets.dataste_support import regist_dataset
from data.datasets.dataset_Base import BaseDataset

__all__ = ['Inference_Dataset_DSM_Fusion_RGB_DSM_CamFeature']

# 5 个相机前缀
CAM_PREFIXES = ["CAM01", "CAM02", "CAM03", "CAM04", "CAM05"]
KEEP_KEYS = ("res2", "res3", "res4", "res5")

# 各尺度的默认 shape (对于 512x512 的 TDOM 图像)
DEFAULT_SHAPES = {
    "res2": (256, 128, 128),
    "res3": (256,  64,  64),
    "res4": (256,  32,  32),
    "res5": (256,  16,  16),
}


@regist_dataset()
class Inference_Dataset_DSM_Fusion_RGB_DSM_CamFeature(BaseDataset):
    """
    每个样本返回:
        cam_features: list[5], 每个元素为 dict{res2/3/4/5: Tensor[C,H,W]}
        cam_exist:    list[5], bool 标记
        xyz_image:    Tensor[3,H,W]
        xyz_dsm:      Tensor[1,H,W]
        xyz_label:    Tensor[H,W]
        name:         str (用于获取 offset)
    """
    def __init__(
        self,
        augmentation: bool,
        cam_feat_data_dir: str,
        xyz_data_dir: str,
        indices_name='indices',
        standardization=True,
        xyz_dsm_none_value=-99999,
        clip=None,
        data_pipeline=None,
        **kwargs
    ):
        super().__init__(
            xyz_data_dir, augmentation, standardization,
            clip, data_pipeline, indices_name,
        )

        self.cam_feat_data_dir = cam_feat_data_dir
        self.xyz_data_dir = xyz_data_dir

        self._image_dir_name = 'image'
        self._label_dir_name = 'label'
        self._dsm_dir_name   = 'dsm'

        if clip is not None:
            self.ids = self.ids[clip[0]:clip[1]]

        self.xyz_dsm_none_value = xyz_dsm_none_value

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = os.path.splitext(self.ids[idx])[0]  # e.g. 6_0000_0360

        # ---------- xyz 图像 ----------
        label_path = os.path.join(self.xyz_data_dir, self._label_dir_name, name + '.tif')
        img_path   = os.path.join(self.xyz_data_dir, self._image_dir_name, name + '.tif')
        dsm_path   = os.path.join(self.xyz_data_dir, self._dsm_dir_name,   name + '.tif')

        xyz_label = self.open(label_path)
        xyz_image = self.open(img_path, channel_sum=3)
        xyz_dsm   = self.open(dsm_path, none_value=self.xyz_dsm_none_value)

        if self.augmentation and self.data_pipeline is not None:
            xyz_image, xyz_label, xyz_dsm = seg_augmentation(
                image=xyz_image,
                mask=[xyz_label, xyz_dsm],
                aug=self.data_pipeline,
            )

        # ---------- 5个相机特征 ----------
        cam_features = []
        cam_exist = []

        for cam_prefix in CAM_PREFIXES:
            feat_path = os.path.join(self.cam_feat_data_dir, f"{cam_prefix}-{name}.pt")

            if os.path.exists(feat_path):
                feat_raw = torch.load(feat_path, map_location='cpu')
                feat_dict = {k: feat_raw[k] for k in KEEP_KEYS if k in feat_raw}
                cam_features.append(feat_dict)
                cam_exist.append(True)
            else:
                # 不存在时填充零
                feat_dict = {
                    k: torch.zeros(shape, dtype=torch.float32)
                    for k, shape in DEFAULT_SHAPES.items()
                }
                cam_features.append(feat_dict)
                cam_exist.append(False)

        # ---------- to tensor ----------
        xyz_image = torch.from_numpy(xyz_image).float()
        xyz_dsm   = torch.from_numpy(xyz_dsm).float()
        xyz_label = torch.from_numpy(xyz_label).long()

        return cam_features, cam_exist, xyz_image, xyz_dsm, xyz_label, name
