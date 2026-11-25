# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import pandas as pd

from data.datasets.dataset_Base import BaseDataset
from data.datasets.data_utils import expand_data_2_BCHW,expand_datalist_2_BCHW
from data.transforms.augmentation import *
from data.datasets.dataste_support import *
from utils.make_samples import *
import torch

__all__ = ['Dataset_DSM_Fusion_image_xyz']

@regist_dataset()
class Dataset_DSM_Fusion_image_xyz(BaseDataset):
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
        self.indices = pd.read_csv(os.path.join(img_data_dir, indices_name+'.csv'))
        # calculate statics: mean value, standard
        self.standardization = standardization

        # self.img_stat_file = self.img_data_dir + '/' + 'stat_binary.txt'
        # self.xyz_stat_file = self.xyz_data_dir + '/' + 'stat_binary.txt'
        # self.img_stat_data=self.read_stat(self.img_stat_file)
        # self.xyz_stat_data=self.read_stat(self.xyz_stat_file)
        
        self.ids = self.indices['name'].tolist()

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
        
    def __len__(self):
        return len(self.views_meta)
    
    def __getitem__(self, idx):
        imgs, labels, grids = [], [], []

        # === 1. 读取 xyz 数据 ===
        image_name = os.path.splitext(os.path.basename(self.views_meta[idx][0]))[0]
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

        return img_tensor, xyz_img_tensor, grid_tensor, label_tensor, xyz_label_tensor

        

    


if __name__=='__main__':

    import pandas as pd
    from data.datasets.data_utils import fusion_collate,dataset_collate_padding
    import torch.utils.data as tdata 
    from osgeo import gdal
    from utils.raster_io import fastly_save_image
    data_pipeline=[{'type' : 'RandomBrightnessContrast' , 'p' : 0.5},
                   {'type': 'Flip' , 'p' : 0.5},
                   {'type' : 'ColorJitter'}]
    dataset=Dataset_DSM_Fusion_image_xyz("/path/to/user/NAS/gwx/DeepL/Melbourne/IMG_MV/train/samples",
                                "/path/to/user/NAS/gwx/DeepL/Melbourne/IMG_MV/train/samples/grid_none",
                                "/path/to/user/NAS/gwx/DeepL/Melbourne/XYZ/train/samples",
                                "/path/to/user/NAS/gwx/DeepL/Melbourne/XYZ_xy_name_map.txt",
                                1,augmentation=False,data_pipeline=data_pipeline,standardization=True,
                                img_dsm_none_value=-99999,xyz_dsm_none_value=-99999,)
                                # ,indices_name='singleview_indices')

    data_loader = tdata.DataLoader(dataset=dataset, batch_size=1,
                                   num_workers=0, shuffle=True,  # cfg.shuffle,
                                   pin_memory=True, collate_fn=dataset_collate_padding)

    for data in dataset:
        img_x,xyz_x,grid,image_label,y2=data
#         if xyz_x is None:
#             continue
#         else:
#             fastly_save_image(img_x,'/model/zhangpl/mvmaper/image1.tif',gdal_type=gdal.GDT_Float32)
#             fastly_save_image(image_label,'/model/zhangpl/mvmaper/label1.tif',gdal_type=gdal.GDT_Float32)
    # batch_iterator=iter(data_loader)
    # for batch in batch_iterator:
    #     img_x,xyz_x,grid,image_label,y2,batch_info=batch
        
        # fastly_save_image(img_x[0].squeeze(0).numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/image.tif',gdal_type=gdal.GDT_Float32)
        # fastly_save_image(xyz_x[0].squeeze(0).numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/image22.tif',gdal_type=gdal.GDT_Float32)
        # fastly_save_image(image_label[0].squeeze(0).numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/label.tif',gdal_type=gdal.GDT_Float32)
        # break
        
#         if len(img_x)==3:# and np.sum(image_label[0].squeeze(0).numpy())/(512*512)>0.4:
#             for i in range(img_x.shape[0]):
#                 fastly_save_image(img_x[i,0:3].numpy()*255,'/path/to/user/NAS/gwx/code/seg_framer_v2/image'+str(i)+'.tif',gdal_type=gdal.GDT_Float32)
#                 fastly_save_image(img_x[i,3:].numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/image_dsm'+str(i)+'.tif',gdal_type=gdal.GDT_Float32)
#                 fastly_save_image(image_label[i].numpy()*255,'/path/to/user/NAS/gwx/code/seg_framer_v2/label'+str(i)+'.tif')
#                 fastly_save_image(np.transpose(grid[i].numpy(),[2,0,1]),
#                                     '/path/to/user/NAS/gwx/code/seg_framer_v2/grid'+str(i)+'.tif',gdal_type=gdal.GDT_Float32)

#             for i in range(y2.shape[0]):
#                 fastly_save_image(y2[i].numpy()*255,'/path/to/user/NAS/gwx/code/seg_framer_v2/xyz_label'+str(i)+'.tif')
#                 fastly_save_image(xyz_x[i,0:3].numpy()*255,'/path/to/user/NAS/gwx/code/seg_framer_v2/xyz'+str(i)+'.tif',gdal_type=gdal.GDT_Float32)
#                 fastly_save_image(xyz_x[i,3:].numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/xyz_dsm'+str(i)+'.tif',gdal_type=gdal.GDT_Float32)
#             break

# if __name__=='__main__':
#     # from data.datasets.data_utils import dataset_collate
#     # import torch.utils.data as tdata 
#     # import gdal
#     # from utils.raster_io import fastly_save_image
#     train_dataset=Dataset_DSM_Fusion_image_xyz(r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\samples_dsm_512_512_512",
#                                 r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\samples_dsm_512_512_512\grid_none",
#                                 r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_XYZ\train\samples_dsm_512_512_512",
#                                 r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\XYZ_xy_name_map.txt",
#                                 1,augmentation=False,standardization=False,dsm_none_value=-99999)#,indices_name='singleview_indices')
#     train_views={}
#     for i in train_dataset.views_meta:
#         view_num=len(i)
#         if view_num in train_views:
#             train_views[view_num]+=view_num
#         else:
#             train_views[view_num]=view_num
#     print(train_views)
#     print(train_views.keys())
    
    # test_dataset=Dataset_DSM_Fusion_image_xyz("/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none",
    #                             "/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none/grid_none",
    #                             "/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/train/samples_none",
    #                             "/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/XYZ_xy_name_map.txt",
    #                             1,augmentation=False,standardization=False)#,indices_name='singleview_indices')
    # test_views={}
    # for i in test_dataset.views_meta:
    #     view_num=len(i)
    #     if view_num in test_views:
    #         test_views[view_num]+=view_num
    #     else:
    #         test_views[view_num]=view_num
