import os
import torch
import numpy as np

from data.transforms.augmentation import seg_augmentation
from data.datasets.dataste_support import regist_dataset
from data.datasets.dataset_Base import BaseDataset


@regist_dataset()
class Dataset_DSM_Fusion_RGB_DSM_ClusterFeature(BaseDataset):
    """
    专门适配 K-means 聚类特征的 Dataset。
    将原本存储在 [K, C, H, W] 中的 K 个聚类特征，拆分为独立的“虚拟视角”列表输出。
    """
    def __init__(self,
                 augmentation: bool,
                 feat_data_dir: str,
                 xyz_data_dir: str,
                 indices_name='indices',
                 standardization=True,
                 xyz_dsm_none_value=-99999,
                 clip=None,
                 data_pipeline=None,
                 max_k=3): # 固定的最大聚类数

        super().__init__(xyz_data_dir, augmentation, standardization,
                         clip, data_pipeline, indices_name)

        self.feat_data_dir = feat_data_dir
        self.xyz_data_dir  = xyz_data_dir
        self.max_k = max_k

        self._image_dir_name = 'image'
        self._label_dir_name = 'label'
        self._dsm_dir_name   = 'dsm'
        self.xyz_dsm_none_value = xyz_dsm_none_value

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = os.path.splitext(self.ids[idx])[0]
        
        # 1. 加载 目标域 (TDOM) 的基础数据 (RGB, DSM, Label)
        label_path = os.path.join(self.xyz_data_dir, self._label_dir_name, name +'.tif')
        img_path   = os.path.join(self.xyz_data_dir, self._image_dir_name, name +'.tif')
        dsm_path   = os.path.join(self.xyz_data_dir, self._dsm_dir_name,   name+'.tif')

        xyz_label = self.open(label_path)
        xyz_image = self.open(img_path, channel_sum=3)
        xyz_dsm   = self.open(dsm_path, none_value=self.xyz_dsm_none_value)

        # 数据增强
        if self.augmentation and self.data_pipeline is not None:
            xyz_image, xyz_label, xyz_dsm = seg_augmentation(
                image=xyz_image,
                mask=[xyz_label, xyz_dsm],
                aug=self.data_pipeline
            )

        # 转换为 Tensor
        xyz_image = torch.from_numpy(xyz_image).float()
        xyz_dsm   = torch.from_numpy(xyz_dsm).float()
        xyz_label = torch.from_numpy(xyz_label).long()

        # 2. 加载 5 个相机 x 3 个聚类 = 15 个视角
        # 根据实际观察，文件都在同一级目录下，命名格式为: 相机名-文件名.pt (如 CAM01-17_0000_0360.pt)
        cam_list = ["CAM01", "CAM02", "CAM03", "CAM04", "CAM05"]
        KEEP_KEYS = ("res2", "res3", "res4", "res5")
        
        all_virtual_features = [] # 最终长度为 15 (5*3)
        all_virtual_exist = []    # 最终长度为 15

        for cam_name in cam_list:
            # 关键修正：路径拼接逻辑改为 {cam_name}-{name}.pt
            feat_path = os.path.join(self.feat_data_dir, f"{cam_name}-{name}.pt")
            
            if os.path.exists(feat_path):
                try:
                    data = torch.load(feat_path, map_location='cpu')
                    clusters = data["clusters"]
                    # 优先读取 2.py 生成的 cluster_valid 掩码
                    valid_flags = data.get("cluster_valid")
                    k_real = clusters["res2"].shape[0]
                except:
                    valid_flags = None
                    k_real = 0
            else:
                # 核心预测：补齐张量必须与真实特征尺度完全一致
                SHAPES = {"res2":(256,128,128), "res3":(256,64,64), "res4":(256,32,32), "res5":(256,16,16)}
                zero_dict = {k: torch.zeros(SHAPES[k]) for k in KEEP_KEYS}
                for _ in range(self.max_k):
                    all_virtual_features.append(zero_dict)
                    all_virtual_exist.append(False)
                continue

            # 每个相机固定产出 3 个视角槽位 (max_k)
            for i in range(self.max_k): 
                view_dict = {}
                # 判定逻辑：如果有 valid_flags 则精准判定，否则按数量判定
                if valid_flags is not None:
                    is_valid = (i < len(valid_flags) and valid_flags[i].item() > 0)
                else:
                    is_valid = (i < k_real)
                
                for scale in KEEP_KEYS:
                    if is_valid:
                        view_dict[scale] = clusters[scale][i]
                    else:
                        # 补齐全 0 特征
                        c, h, w = {"res2":(256,128,128), "res3":(256,64,64), "res4":(256,32,32), "res5":(256,16,16)}[scale]
                        view_dict[scale] = torch.zeros((c, h, w), dtype=torch.float32)
                
                all_virtual_features.append(view_dict)
                all_virtual_exist.append(is_valid)

        # 转换 exist 为 Tensor [15]
        all_virtual_exist = torch.tensor(all_virtual_exist, dtype=torch.bool)

        # 返回 6 个参数：特征、掩码、RGB、DSM、标签、文件名
        return all_virtual_features, all_virtual_exist, xyz_image, xyz_dsm, xyz_label, name
        # return all_virtual_features, all_virtual_exist, xyz_image, xyz_dsm, xyz_label
