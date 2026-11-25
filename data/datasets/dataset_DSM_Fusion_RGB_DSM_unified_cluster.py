import os
import torch
import numpy as np

from data.transforms.augmentation import seg_augmentation
from data.datasets.dataste_support import regist_dataset
from data.datasets.dataset_Base import BaseDataset


@regist_dataset()
class Dataset_DSM_Fusion_RGB_DSM_UnifiedCluster(BaseDataset):
    """
    适配单一 PT 文件的聚类特征 Dataset。
    每个 PT 文件包含该 Tile 的所有聚类特征，将其拆分为独立的“虚拟视角”列表输出。
    文件名格式: {xyz_key}.pt
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
                 max_k=15,
                 shuffle_views=False): # 默认为 15，适配之前的 5x3 结构

        super().__init__(xyz_data_dir, augmentation, standardization,
                         clip, data_pipeline, indices_name)

        self.feat_data_dir = feat_data_dir
        self.xyz_data_dir  = xyz_data_dir
        self.max_k = max_k
        self.shuffle_views = shuffle_views

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

        # 2. 加载单一 PT 文件，每个 cluster 作为一个“虚拟视角”
        feat_path = os.path.join(self.feat_data_dir, f"{name}.pt")
        KEEP_KEYS = ("res2", "res3", "res4", "res5")
        SHAPES = {"res2":(256,128,128), "res3":(256,64,64), "res4":(256,32,32), "res5":(256,16,16)}
        
        all_virtual_features = [] 
        all_virtual_exist = []    

        if os.path.exists(feat_path):
            try:
                data = torch.load(feat_path, map_location='cpu')
                if "clusters" in data:
                    clusters = data["clusters"]
                    cluster_valid = data.get("cluster_valid") # [K]
                else:
                    clusters = {}
                    for scale in KEEP_KEYS:
                        if scale not in data:
                            raise KeyError(scale)
                        feat = data[scale]
                        if feat.ndim == 3:
                            feat = feat.unsqueeze(0)
                        clusters[scale] = feat
                    cluster_valid = torch.ones(clusters["res2"].shape[0], dtype=torch.bool)
                K = clusters["res2"].shape[0]
            except Exception:
                K = 0
                clusters = None
                cluster_valid = None
        else:
            K = 0
            clusters = None
            cluster_valid = None

        # 遍历 max_k 个槽位
        for i in range(self.max_k):
            view_dict = {}
            is_valid = False
            
            if i < K:
                # 在 K 范围内，优先使用 cluster_valid 判定
                if cluster_valid is not None:
                    is_valid = bool(cluster_valid[i].item() > 0)
                else:
                    is_valid = True
                
                for scale in KEEP_KEYS:
                    if is_valid:
                        view_dict[scale] = clusters[scale][i]
                    else:
                        c, h, w = SHAPES[scale]
                        view_dict[scale] = torch.zeros((c, h, w), dtype=torch.float32)
            else:
                # 超过 K 的部分，补齐 0 tensor，标记为无效
                is_valid = False
                for scale in KEEP_KEYS:
                    c, h, w = SHAPES[scale]
                    view_dict[scale] = torch.zeros((c, h, w), dtype=torch.float32)
            
            all_virtual_features.append(view_dict)
            all_virtual_exist.append(is_valid)

        # 转换 exist 为 Tensor [max_k]
        all_virtual_exist = torch.tensor(all_virtual_exist, dtype=torch.bool)

        if self.shuffle_views and self.max_k > 1:
            perm = torch.randperm(self.max_k)
            all_virtual_features = [all_virtual_features[i] for i in perm.tolist()]
            all_virtual_exist = all_virtual_exist[perm]

        # 返回 6 个参数：特征、掩码、RGB、DSM、标签、文件名
        return all_virtual_features, all_virtual_exist, xyz_image, xyz_dsm, xyz_label, name
