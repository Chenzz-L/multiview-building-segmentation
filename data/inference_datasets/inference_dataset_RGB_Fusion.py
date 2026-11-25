import os

import numpy as np

from data.inference_datasets.inference_dataset_Base import InferenceBaseDataset
from data.datasets.dataste_support import *
from utils.raster_io import *
from utils.make_samples import *
from utils.make_samples import read_crop_corner_file


__all__=['InferenceDataset_RGB_Fusion']

@regist_dataset()
class InferenceDataset_RGB_Fusion(InferenceBaseDataset):
    def __init__(self,
                 data_dir: str,
                 xyz_label_dir: str,
                 xyz_img_pair_file: str,
                 corner_coordination_files_dir: str,
                 points_pair_dir:str,
                 dsm_dir:str,
                 standardization: bool = True,
                 img_type: str = 'tif'):
        """测试过程中的DOM Datset

        Args:
            data_dir (str): 模型输入数据的根目录
            xyz_label_dir (str): 物方标签路径
            xyz_img_pair_file (str): 物方和像方对应关系文件路径
            corner_coordination_files_dir (str): 预先的切图角点路径
            points_pair_dir (str): 像点对路径
            dsm_dir (str): dsm文件路径
            standardization (bool, optional): 是否进行标准化. Defaults to True.
            img_type (str, optional): 输入文件的后缀. Defaults to 'tif'.
        """
        super(InferenceDataset_RGB_Fusion, self).__init__(data_dir, standardization, img_type)
        self.xyz_label_dir = xyz_label_dir
        self._dsm_dir = dsm_dir
        self.xyz_img_pair_file = xyz_img_pair_file
        self.points_pair_dir=points_pair_dir
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
        if self.standardization:
            if self.stat_data is None:
                raise Exception("statics data has None value")
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)
        imgs = []
        label_datas = []
        points_pair_datas = []

        # read xyz data
        model_predict_name = os.path.splitext(os.path.basename(self.views_meta[idx][0]))[0]
        xyz_num = model_predict_name.split("_")[-1]

        xyz_label_file = os.path.join(self.xyz_label_dir, self.xyz_img_dict[xyz_num] + ".tif")
        xyz_label_data=self.open(xyz_label_file)

        # read image
        for j, file in enumerate(self.views_meta[idx]):
            # read image
            image_data=self.open(os.path.join(self._image_dir, '%s' % file),
                            channel_sum=3,
                            standardization=self.standardization,mean=0,std=255.0)
            
            # read label
            label_data=self.open(os.path.join(self._label_dir, '%s' % file))
            
            # read point pair
            points_pair_data=self.open(os.path.join(self.points_pair_dir,'%s' % file))
            
            # read point dsm
            dsm_data=self.open(os.path.join(self._dsm_dir,'%s' % file))

            points_pair_data = np.concatenate([dsm_data, points_pair_data], axis=0)
            points_pair_datas.append(points_pair_data)
            imgs.append(image_data)
            label_datas.append(label_data)

        img_out_data = {"image": imgs, "label": label_datas,
                        'points_pair':points_pair_datas,
                        'xyz_label_data' : xyz_label_data}
        info = {"batch_info": [len(imgs)],
                "name": self.views_meta[idx] + [self.xyz_img_dict[xyz_num] + ".tif"],
                "corners":self.corner_coordination_list[idx]}
        return img_out_data, info
