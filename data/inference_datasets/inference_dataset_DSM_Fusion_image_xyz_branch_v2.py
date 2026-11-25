# -*- coding: utf-8 -*-
import os
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.utils.data as tdata
import numpy as np
import pandas as pd
from osgeo import gdal
from data.transforms.augmentation import *
from data.datasets.data_utils import expand_data_2_BCHW,expand_datalist_2_BCHW
from data.inference_datasets.inference_dataset_Base import InferenceBaseDataset
from data.datasets.dataste_support import *
from utils.make_samples import *
from utils.make_samples import read_crop_corner_file
from utils.raster_io import fastly_save_image
import torch

__all__ = ['InferenceDataset_DSM_Fusion_image_xyz_v2']

@regist_dataset()
class InferenceDataset_DSM_Fusion_image_xyz_v2(InferenceBaseDataset):
    def __init__(self,
                 img_data_dir: str,
                 grid_dir: str,
                 xyz_data_dir: str,
                 xyz_img_pair_file: str,
                 grid_scale:float,
                 augmentation: bool = False,
                 standardization: bool = True,
                 img_dsm_none_value: float = 0,
                 xyz_dsm_none_value: float = 0,
                 mask=False,
                 clip: list = None,
                 data_pipeline=None,
                 indices_name='indices'):
        assert os.path.exists(img_data_dir)
        assert os.path.exists(xyz_data_dir)
        assert os.path.exists(grid_dir)
        self.img_data_dir = img_data_dir
        self.xyz_data_dir=xyz_data_dir
        self.gird_dir = grid_dir
        self._image_dir_name = 'image'
        self._label_dir_name = 'label'
        self._dsm_dir_name = 'dsm'
        
        self.augmentation = augmentation
        # get indices, return pd.DataFrame, a python dictionary
        # self.indices = pd.read_csv(os.path.join(img_data_dir, 'indices.csv'))
        if indices_name!=None:
            with open(os.path.join(img_data_dir,indices_name + '.txt'),'r') as f:
                outs=f.readlines()
                self.ids=[out.rstrip() for out in outs]
        else:
            self.ids = [os.path.basename(img) for img in glob.glob(os.path.join(img_data_dir,self._image_dir_name,"*.tif"))]
        # calculate statics: mean value, standard
        self.standardization = standardization

        # self.img_stat_file = self.img_data_dir + '/' + 'stat_binary.txt'
        # self.xyz_stat_file = self.xyz_data_dir + '/' + 'stat_binary.txt'
        # self.img_stat_data=self.read_stat(self.img_stat_file)
        # self.xyz_stat_data=self.read_stat(self.xyz_stat_file)
        
        # self.ids = self.indices['name'].tolist()

        self.img_dsm_none_value = img_dsm_none_value
        self.xyz_dsm_none_value = xyz_dsm_none_value
        self.mask = mask
        # self.pair_file=pair_file
        self.views_meta = self.build_pair()

        self.xyz_img_dict = self.read_xyz_img_pair_file(xyz_img_pair_file)
        self.grid_scale = grid_scale
        if clip is not None:
            assert len(clip) == 2
            self.views_meta = self.views_meta[clip[0]:clip[1]]

        if augmentation and data_pipeline:
            if not 'ToFloat' in data_pipeline[0].values():# 自定义ToFloat
                data_pipeline.insert(0,{'type':'ToFloat','max_value':255})
            self.data_pipeline=build_agument_pipeline(data_pipeline,True)
            self.augmentation = augmentation
        else:
            self.augmentation=False

    def build_pair(self):
        metas = make_pair_with_indices(self.ids, 2)
        return metas

    def read_xyz_img_pair_file(self, xyz_img_pair_file):
        with open(xyz_img_pair_file, "r") as f:
            scans = f.readlines()
            scans = [line.replace("\t", " ").strip().split(" ") for line in scans]
            scans = np.array(scans)
            scans_dict = dict(zip(scans[:, 1], scans[:, 0]))
        return scans_dict

    def __len__(self):
        return len(self.views_meta)

    def __getitem__(self, idx):
        imgs, labels, grids = [], [], []

        # === 1. 读取 xyz 数据 ===
        image_name = os.path.splitext(os.path.basename(self.views_meta[idx][0]))[0]
        
        xyz_num = image_name.split("_")[-3].split("-")[-1]
        num = "_".join(image_name.split("_")[-2:])
        xyz_suffix = f"{self.xyz_img_dict[xyz_num]}_{num}.tif"
        xyz_image_name = xyz_suffix
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

        # === 2. 读取多视角影像 ===
        for file in self.views_meta[idx]:
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
        else:
            print(f"[WARN] grid_data shape unexpected: {grid_data.shape}")

        # === 4. 转 Tensor ===
        img_tensor = torch.from_numpy(img.copy()).float()
        xyz_img_tensor = torch.from_numpy(xyz_image_data.copy()).float()
        grid_tensor = torch.from_numpy(grid_data.copy()).float()
        label_tensor = torch.from_numpy(label_data.copy()).long()
        xyz_label_tensor = torch.from_numpy(xyz_label_data.copy()).long()

        return img_tensor, xyz_img_tensor, grid_tensor, label_tensor, xyz_label_tensor,xyz_image_name

# if __name__=='__main__':

#     dataset=InferenceDataset_DSM_Fusion_image_xyz(r"/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/test",
#                                             r"/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/test",
#                                             r"/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/XYZ_xy_name_map.txt",
#                                             r"/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/test/test_cornertxts/",
#                                             r"/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/test/point_pairs/",
#                                             True)

#     data_loader = tdata.DataLoader(dataset=dataset, batch_size=1,
#                                    num_workers=0, shuffle=True,  # cfg.shuffle,
#                                    pin_memory=True, collate_fn=lambda x:x)


#     batch_iterator=iter(data_loader)
#     for batch in batch_iterator:
#         data,info=batch[0]
#         fastly_save_image(data['image'][0],'/path/to/user/NAS/gwx/code/seg_framer_v2/image.tif',gdal_type=gdal.GDT_Float32)
#         fastly_save_image(data['xyz_image_data'],'/path/to/user/NAS/gwx/code/seg_framer_v2/image22.tif',gdal_type=gdal.GDT_Float32)
#         print('ok')
#         break
