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


__all__ = ['Dataset_RGB_Fusion_image_xyz']

# @regist_dataset()
class Dataset_RGB_Fusion_image_xyz(BaseDataset):
    def __init__(self,
                 img_data_dir: str,
                 grid_dir: str,
                 xyz_data_dir: str,
                 xyz_img_pair_file: str,
                 grid_scale:float,
                 augmentation: bool = False,
                 standardization: bool = True,
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
        self.indices = pd.read_csv(os.path.join(img_data_dir, indices_name+'.csv'))

        self.standardization = standardization

        self.img_stat_file = self.img_data_dir + '/' + 'stat_binary.txt'
        self.xyz_stat_file = self.xyz_data_dir + '/' + 'stat_binary.txt'
        self.img_stat_data=self.read_stat(self.img_stat_file)
        self.xyz_stat_data=self.read_stat(self.xyz_stat_file)
        
        self.ids = self.indices['name'].tolist()

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
        imgs = []
        label_datas = []
        grid_datas = []

        # read xyz data
        image_name = os.path.splitext(os.path.basename(self.views_meta[idx][0]))[0]
        xyz_num = image_name.split("_")[-3]
        num = image_name.split("_")[-2:]
        num = "_".join(num)
        xyz_suffix=self.xyz_img_dict[xyz_num] + "_" + num + ".tif"
        file = os.path.join(self.xyz_data_dir, self._label_dir_name,xyz_suffix)
        xyz_label_data=self.open(file)
        
        file = os.path.join(self.xyz_data_dir, self._image_dir_name,xyz_suffix)
        xyz_image_data=self.open(file,channel_sum=3,
                                 standardization=not self.augmentation,mean=0,std=255.0)
        
        if self.augmentation and self.data_pipeline!=None:
            xyz_image_data=seg_augmentation_imageonly(xyz_image_data,self.data_pipeline)
        
        file = os.path.join(self.xyz_data_dir, self._dsm_dir_name,xyz_suffix)

        
        for j, file in enumerate(self.views_meta[idx]):
            
            # read image
            image_data=self.open(os.path.join(self.img_data_dir,self._image_dir_name, '%s' % file),
                            channel_sum=3,
                            standardization=not self.augmentation,mean=0,std=255.0)
            
            if self.augmentation and self.data_pipeline!=None:
                image_data=seg_augmentation_imageonly(image_data,self.data_pipeline)

            # read label
            label_data=self.open(os.path.join(self.img_data_dir,self._label_dir_name, '%s' % file))

            # read grid
            grid_data=self.open(os.path.join(self.gird_dir, file),scale=self.grid_scale)
            
            grid_datas.append(grid_data)
            imgs.append(image_data)
            label_datas.append(label_data)


        expand_datalist_2_BCHW(imgs)
        expand_datalist_2_BCHW(label_datas)
        expand_datalist_2_BCHW(grid_datas)
        xyz_label_data = expand_data_2_BCHW(xyz_label_data)
        xyz_image_data = expand_data_2_BCHW(xyz_image_data)
        

        img=np.concatenate(imgs,axis=0)
        grid_data = np.concatenate(grid_datas,axis=0)
        label_data = np.concatenate(label_datas, axis=0)
        grid_data = grid_data.transpose((0, 2, 3, 1))
        return img,xyz_image_data, grid_data, label_data, xyz_label_data


if __name__=='__main__':

    import pandas as pd
    from data.datasets.data_utils import fusion_collate
    import torch.utils.data as tdata 
    import gdal
    from utils.raster_io import fastly_save_image
    # data_pipeline=[{'type' : 'RandomBrightnessContrast' , 'p' : 0.5},
    #                {'type': 'Flip' , 'p' : 0.5},
    #                {'type' : 'ColorJitter'}]
    dataset=Dataset_RGB_Fusion_image_xyz("/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none",
                                "/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none/grid_none",
                                "/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/train/samples_none",
                                "/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/XYZ_xy_name_map.txt",
                                0.25,augmentation=True,data_pipeline=None)

    data_loader = tdata.DataLoader(dataset=dataset, batch_size=1,
                                   num_workers=0, shuffle=True,  # cfg.shuffle,
                                   pin_memory=True, collate_fn=fusion_collate)
    
    out_dir=r'/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/3views_image'
    batch_iterator=iter(data_loader)
    n=0
    for batch in batch_iterator:
        img_x,xyz_x,grid,image_label,y2,batch_info=batch
        if img_x.shape[0]==3:
            if not os.path.exists(os.path.join(out_dir,str(n))):
                os.makedirs(os.path.join(out_dir,str(n)))
            for i in range(img_x.shape[0]):
                out=img_x[i].numpy()
                fastly_save_image(out,os.path.join(out_dir,str(n),str(i)+".tif"),gdal_type=gdal.GDT_Float32)
            
            fastly_save_image(xyz_x[0].numpy(),os.path.join(out_dir,str(n),"xyz.tif"),gdal_type=gdal.GDT_Float32)
            n+=1
