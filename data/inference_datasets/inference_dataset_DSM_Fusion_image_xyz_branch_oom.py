# -*- coding: utf-8 -*-
import os
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from typing import Optional

import torch.utils.data as tdata
import numpy as np
import pandas as pd
from osgeo import gdal

from data.inference_datasets.inference_dataset_Base import InferenceBaseDataset
from data.datasets.dataste_support import *
from utils.make_samples import *
from utils.make_samples import read_crop_corner_file

__all__ = ['InferenceDataset_DSM_Fusion_image_xyz_oom']
  
@regist_dataset()
class InferenceDataset_DSM_Fusion_image_xyz_oom(InferenceBaseDataset):
    def __init__(self,
                 img_data_dir: str,
                 xyz_data_dir:str,
                 xyz_img_pair_file: str,
                 corner_coordination_files_dir: str,
                 points_pair_dir:str,
                 standardization: bool = True,
                 img_dsm_none_value: float = 0,
                 xyz_dsm_none_value: float = 0,
                 mask:bool=False,
                 indices_name=None):
        assert os.path.exists(img_data_dir)
        assert os.path.exists(xyz_data_dir)
        self.img_data_dir = img_data_dir
        self.xyz_data_dir=xyz_data_dir
        self.xyz_img_pair_file = xyz_img_pair_file
        self.points_pair_dir=points_pair_dir
        self._image_dir_name = 'image'
        self._label_dir_name = 'label'
        self._dsm_dir_name = 'dsm'
        
        if indices_name!=None:
            with open(os.path.join(img_data_dir,indices_name+'.txt'),'r') as f:
                outs=f.readlines()
                self.ids=[out.rstrip() for out in outs]
        else:
            self.ids = [os.path.basename(img) for img in glob.glob(os.path.join(img_data_dir,self._image_dir_name,"*.tif"))]
        # calculate statics: mean value, standard
        self.standardization = standardization

        self.img_stat_file = self.img_data_dir + '/' + 'stat_binary.txt'
        self.xyz_stat_file = self.xyz_data_dir + '/' + 'stat_binary.txt'
        self.img_stat_data=self.read_stat(self.img_stat_file)
        self.xyz_stat_data=self.read_stat(self.xyz_stat_file)
        
        self.img_dsm_none_value = img_dsm_none_value
        self.xyz_dsm_none_value = xyz_dsm_none_value
        self.mask=mask
        self.views_meta = self.build_pair()
        self.xyz_img_dict = self.read_xyz_img_pair_file(xyz_img_pair_file)
        self.corner_coordination_list = []
        for file_names in self.views_meta:
            _=[]
            for file_name in file_names:
                t_file_name=file_name.split(".")[0]
                corner_coordination_file = os.path.join(corner_coordination_files_dir, t_file_name+".txt")
                _.append(read_crop_corner_file(corner_coordination_file))
            self.corner_coordination_list.append(_)

    def build_pair(self):
        metas = make_pair_with_indices(self.ids, 1)
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
    
    def read_img_data(self,img_file,dsm_file,point_pair_file,label_file,
                      xoff:Optional[int]=None,yoff:Optional[int]=None,
                      w:Optional[int]=None,h:Optional[int]=None):
        image_data=self.open(img_file,
                            channel_sum=3,
                            standardization=self.standardization,
                            mean=0,std=255.0,
                            xoff=xoff,yoff=yoff,w=w,h=h)
        # read dsm，需要dsm进行仿射变换拟合，因此这里不能进行任何操作
        dsm_data=self.open(dsm_file,
                            mask=self.mask,
                            mask_value=self.img_dsm_none_value,
                            xoff=xoff,yoff=yoff,w=w,h=h)   

        # read points_pair
        if point_pair_file!=None:
            points_pair_data=self.open(point_pair_file,xoff=xoff,yoff=yoff,w=w,h=h)
            points_pair_data = np.concatenate([dsm_data[0:1,:,:].copy(), points_pair_data], axis=0)
        else:
            points_pair_data=None
            
        if self.img_dsm_none_value is not None and self.img_dsm_none_value!=0:
            dsm_data[dsm_data == self.img_dsm_none_value] = 0
        
        if self.standardization:
            dsm_data = (dsm_data / self.img_stat_data[-1][-2]).astype('float32')
            
        image_data = np.concatenate((image_data, dsm_data), axis=0)

        # read label
        if label_file!=None:
            label_data=self.open(label_file,xoff=xoff,yoff=yoff,w=w,h=h)
        else:
            label_data=None
        
        return image_data,points_pair_data,label_data
    
    def read_xyz_data(self,xyz_image_file,xyz_dsm_file,xyz_label_file):
        xyz_label_data=self.open(xyz_label_file)

        xyz_image_data=self.open(xyz_image_file,channel_sum=3,
                                 standardization=self.standardization,mean=0,std=255.0)
        
        xyz_dsm_data=self.open(xyz_dsm_file,
                           standardization=self.standardization,
                           mean=0,
                           std=self.xyz_stat_data[-1][-2],
                           mask=self.mask,
                           mask_value=self.xyz_dsm_none_value,
                           none_value=self.xyz_dsm_none_value)
        
        xyz_image_data = np.concatenate((xyz_image_data, xyz_dsm_data), axis=0)

        return xyz_image_data,xyz_label_data
        
    def __getitem__(self, idx):
        imgs = []
        dsms = []
        label_datas = []
        points_pair_datas=[]
        
        # read xyz data
        image_name = os.path.splitext(os.path.basename(self.views_meta[idx][0]))[0]
        xyz_num = image_name.split("_")[-1]
        suffix=self.xyz_img_dict[xyz_num] + ".tif"
        
        xyz_label_file = os.path.join(self.xyz_data_dir,self._label_dir_name,suffix)

        xyz_image_file = os.path.join(self.xyz_data_dir, self._image_dir_name,suffix)
        
        xyz_dsm_file = os.path.join(self.xyz_data_dir, self._dsm_dir_name,suffix)

        # read image
        for j, file in enumerate(self.views_meta[idx]):
            points_pair_datas.append(os.path.join(self.points_pair_dir,'%s' % file))
            imgs.append(os.path.join(self.img_data_dir,self._image_dir_name, '%s' % file))
            dsms.append(os.path.join(self.img_data_dir,self._dsm_dir_name,'%s' % file))
            label_datas.append(os.path.join(self.img_data_dir,self._label_dir_name, '%s' % file))

        img_out_data = {"image_files": imgs, "dsm_files": dsms,"label_files": label_datas,
                        'points_pair_files':points_pair_datas,
                        'xyz_label_files' : xyz_label_file,
                        'xyz_image_files':xyz_image_file,
                        'xyz_dsm_files':xyz_dsm_file,
                        "corners":self.corner_coordination_list[idx]}
        return img_out_data
