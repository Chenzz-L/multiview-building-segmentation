# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
# import torch
# import torch.utils.data as tdata
# from osgeo import gdal

from data.datasets.dataset_Base import BaseDataset
from data.datasets.data_utils import expand_data_2_BCHW,expand_datalist_2_BCHW
from data.transforms.augmentation import *
from data.datasets.dataste_support import *
from utils.make_samples import *


__all__ = ['Dataset_DSM_Fusion']

@regist_dataset()
class Dataset_DSM_Fusion(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 grid_dir: str,
                 xyz_label_dir: str,
                 xyz_img_pair_file: str,
                 grid_scale:float,
                 augmentation: bool = False,
                 standardization: bool = True,
                 dsm_none_value: float = 0,
                 mask=False,
                 clip: list = None,
                 data_pipeline=None,
                 indices_name=None):
        r"""
        :param dsm_none_value: no data value in dsm, usually -99999 or -32767, need to change into 0
        usage:
        train_set = Roof_DSM_Dataset(data_dir=trainset_dir)
        # directly set statics from outside
        train_set.set_stat(rgb_mean, rgb_std, dsm_stat)
        # read stat if statics not given
            if os.path.exists(txt_file):
                train_set.read_stat(txt_file)
        # calculate stat
        train_set.calculate_stat()
        # balance_samples
        self.add_positive_samples(positive_ratio_thresh=0.2, balance_rate=0.7, str_pr='positive_ratio')
        """
        super().__init__(data_dir, augmentation, standardization,None,None,indices_name)
        self._dsm_dir = os.path.join(data_dir, 'dsm')
        self.dsm_none_value = dsm_none_value
        self.mask = mask
        # self.pair_file=pair_file
        self.views_meta = self.build_pair()
        self.gird_dir = grid_dir
        self.xyz_label_dir = xyz_label_dir
        self.xyz_img_dict = self.read_xyz_img_pair_file(xyz_img_pair_file)
        self.grid_scale = grid_scale
        if clip is not None:
            assert len(clip) == 2
            self.views_meta = self.views_meta[clip[0]:clip[1]]

        if augmentation and data_pipeline:
            if not 'ToFloat' in data_pipeline[0].values():# 自定义ToFloat
                data_pipeline.insert(0,{'type':'ToFloat','max_value':255})
            self.data_pipeline=build_agument_pipeline(data_pipeline,True)
            self.augmentation=augmentation
        
    def build_pair(self):
        metas = make_pair_with_indices(self.ids)
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
        if self.standardization:
            if self.stat_data is None :
                raise Exception("statics data has None value")
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)
        assert os.path.isdir(self._dsm_dir)
        imgs = []
        label_datas = []
        grid_datas = []

        # read xyz data
        model_predict_name = os.path.splitext(os.path.basename(self.views_meta[idx][0]))[0]
        xyz_num = model_predict_name.split("_")[-3]
        num = model_predict_name.split("_")[-2:]
        num = "_".join(num)
        xyz_label_file = os.path.join(self.xyz_label_dir, self.xyz_img_dict[xyz_num] + "_" + num + ".tif")
        
        xyz_label_data=self.open(xyz_label_file)
        
        
        for j, file in enumerate(self.views_meta[idx]):
            
            # read image
            image_data=self.open(os.path.join(self._image_dir, '%s' % file),
                            channel_sum=3,
                            standardization=not self.augmentation,mean=0,std=255.0)
            
            if self.augmentation and self.data_pipeline!=None:
                image_data=seg_augmentation_imageonly(image_data,self.data_pipeline)
            
            # read dsm data
            dsm_data=self.open(os.path.join(self._dsm_dir, '%s' % file),
                                standardization=self.standardization,mean=0,std=self.stat_data[-1][-2],
                                mask=self.mask,mask_value=self.dsm_none_value,none_value=self.dsm_none_value)

            image_data = np.concatenate((image_data, dsm_data), axis=0)

            # read label
            label_data=self.open(os.path.join(self._label_dir, '%s' % file))

            # read grid
            grid_data=self.open(os.path.join(self.gird_dir, file),scale=self.grid_scale)
            
            grid_datas.append(grid_data)
            imgs.append(image_data)
            label_datas.append(label_data)

        # if self.augmentation:
            # xyz_label_data, img_labels = data_augumentation_group_views(xyz_label_data, imgs, label_datas)
            # xyz_label_data,grid_datas,img_labels=data_augumentation_group_views(xyz_label_data,grid_datas,imgs,label_datas)
            # imgs, label_datas = img_labels
            
        expand_datalist_2_BCHW(imgs)
        expand_datalist_2_BCHW(label_datas)
        expand_datalist_2_BCHW(grid_datas)
        xyz_label_data = expand_data_2_BCHW(xyz_label_data)

        img=np.concatenate(imgs,axis=0)
        grid_data = np.concatenate(grid_datas,axis=0)
        label_data = np.concatenate(label_datas, axis=0)
        grid_data = grid_data.transpose((0, 2, 3, 1))
        return img, grid_data, label_data, xyz_label_data


# if __name__=='__main__':

#     import pandas as pd
#     from data.datasets.data_utils import dataset_collate
#     import torch.utils.data as tdata 
#     import gdal
#     from utils.raster_io import fastly_save_image
#     # his={}
#     # indices=pd.read_csv(os.path.join('/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/test/samples_none', 'indices.csv'))
#     # ids=indices['name'].tolist()
#     # metas = make_pair_with_indices(ids)
#     # for i in metas:
#     #     group_num=len(i)
#     #     if str(group_num) in his:
#     #         his[str(group_num)]+=1
#     #     else:
#     #         his[str(group_num)]=1
#     data_pipeline=[{'type' : 'RandomBrightnessContrast' , 'p' : 0.5},
#                    {'type': 'Flip' , 'p' : 0.5},
#                    {'type' : 'ColorJitter'}]
#     dataset=Dataset_DSM_Fusion("/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none",
#                                 "/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none/grid_none",
#                                 "/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/train/samples_none/label",
#                                 "/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/XYZ_xy_name_map.txt",
#                                 0.25,augmentation=True,data_pipeline=data_pipeline)

#     data_loader = tdata.DataLoader(dataset=dataset, batch_size=4,
#                                    num_workers=0, shuffle=True,  # cfg.shuffle,
#                                    pin_memory=True, collate_fn=dataset_collate)


#     batch_iterator=iter(data_loader)
#     for batch in batch_iterator:
#         x,grid,image_label,y2,batch_info=batch
#         fastly_save_image(x[0].squeeze(0).numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/image.tif',gdal_type=gdal.GDT_Float32)
#         fastly_save_image(image_label[0].squeeze(0).numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/label.tif',gdal_type=gdal.GDT_Float32)
#         break
        # for i in range(x.shape[0]):
            # out=x[i].numpy()
            # fastly_save_image(out,os.path.join(r"/path/to/project/Users\35416\Desktop\check_data\dataset\image",str(i)+".tif"),gdal_type=gdal.GDT_Float32)
            # fastly_save_image(y1[i].numpy(),os.path.join(r"/path/to/project/Users\35416\Desktop\check_data\dataset\label1",str(i)+".tif"),gdal_type=gdal.GDT_Float32)
            # fastly_save_image(np.transpose(grid[i].numpy(),[2,0,1]),
            #                   os.path.join(r"/path/to/project/Users\35416\Desktop\check_data\dataset\grid",str(i)+".tif"),gdal_type=gdal.GDT_Float32)

        # for i in range(y2.shape[0]):
        #     fastly_save_image(y2[i].numpy(), os.path.join(r"/path/to/project/Users\35416\Desktop\check_data\dataset\label2", str(i) + ".tif"),
        #                       gdal_type=gdal.GDT_Float32)
        # break
