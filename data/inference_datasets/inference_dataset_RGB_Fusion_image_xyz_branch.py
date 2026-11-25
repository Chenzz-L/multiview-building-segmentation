# -*- coding: utf-8 -*-
import os
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.utils.data as tdata
import numpy as np
import pandas as pd
from osgeo import gdal

from data.inference_datasets.inference_dataset_Base import InferenceBaseDataset
from data.datasets.dataste_support import *
from utils.make_samples import *
from utils.make_samples import read_crop_corner_file


__all__ = ['InferenceDataset_RGB_Fusion_image_xyz']

@regist_dataset()
class InferenceDataset_RGB_Fusion_image_xyz(InferenceBaseDataset):
    def __init__(self,
                 img_data_dir: str,
                 xyz_data_dir:str,
                 xyz_img_pair_file: str,
                 corner_coordination_files_dir: str,
                 points_pair_dir:str,
                 standardization: bool = True,
                 indices_name=None):
        assert os.path.exists(img_data_dir)
        assert os.path.exists(xyz_data_dir)
        self.img_data_dir = img_data_dir
        self.xyz_data_dir=xyz_data_dir
        self.xyz_img_pair_file = xyz_img_pair_file
        self.points_pair_dir=points_pair_dir
        self._image_dir_name = 'image'
        self._label_dir_name = 'label'
        self._dsm_dir_name='dsm'

        
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

    def __getitem__(self, idx):
        imgs = []
        label_datas = []
        points_pair_datas=[]
        
        # read xyz data
        image_name = os.path.splitext(os.path.basename(self.views_meta[idx][0]))[0]
        xyz_num = image_name.split("_")[-1]
        suffix=self.xyz_img_dict[xyz_num] + ".tif"
        
        file = os.path.join(self.xyz_data_dir,self._label_dir_name,suffix)
        xyz_label_data=self.open(file)

        file = os.path.join(self.xyz_data_dir, self._image_dir_name,suffix)
        xyz_image_data=self.open(file,channel_sum=3,
                                 standardization=self.standardization,mean=0,std=255.0)

        # read image
        for j, file in enumerate(self.views_meta[idx]):
            # read image
            image_data=self.open(os.path.join(self.img_data_dir,self._image_dir_name, '%s' % file),
                            channel_sum=3,
                            standardization=self.standardization,mean=0,std=255.0)
            
            #read points_pair
            points_pair_data=self.open(os.path.join(self.points_pair_dir,'%s' % file))

            # 为仿射变换提供DSM数据
            dsm_data=self.open(os.path.join(self.img_data_dir,self._dsm_dir_name,'%s' % file))   

            points_pair_data = np.concatenate([dsm_data, points_pair_data], axis=0)

            # read label
            label_data=self.open(os.path.join(self.img_data_dir,self._label_dir_name, '%s' % file))

            points_pair_datas.append(points_pair_data)
            imgs.append(image_data)
            label_datas.append(label_data)

        img_out_data = {"image": imgs, "label": label_datas,
                        'points_pair':points_pair_datas,
                        'xyz_label_data' : xyz_label_data,
                        'xyz_image_data':xyz_image_data}
        info = {"batch_info": [len(imgs)], 
                "name": self.views_meta[idx] + [self.xyz_img_dict[xyz_num] + ".tif"],
                "corners":self.corner_coordination_list[idx]}
        return img_out_data, info


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
