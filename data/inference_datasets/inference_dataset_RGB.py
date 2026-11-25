# -*- coding: utf-8 -*-
import os

from data.inference_datasets.inference_dataset_Base import InferenceBaseDataset
from data.datasets.dataste_support import *

__all__ = ['InferenceDataset_RGB']

@regist_dataset()
class InferenceDataset_RGB(InferenceBaseDataset):
    """
        与训练dataset不同的是，label_data不需要添加一个维度
    """
    def __init__(self,
                 data_dir: str,
                 standardization: bool = True,
                 mean: list = None,
                 std: list = None,
                 img_type:str='tif'):
        super().__init__(data_dir,standardization,mean,std,img_type)


    def __getitem__(self, idx):
        if self.standardization:
            mean = self.mean
            std = self.std
        assert os.path.isdir(self._image_dir)

        # read image
        # image_data=self.open(os.path.join(self._image_dir, self.ids[idx]),
        #                     channel_sum=3,
        #                     standardization=self.standardization,mean=0,std=255.0)
        image_data=self.open(os.path.join(self._image_dir, self.ids[idx]),
                             channel_sum=3)
        out_data = {"image": image_data, "name": self.ids[idx]}
        # read label
        if self._label_dir is not None:
            label_data=self.open(os.path.join(self._label_dir, self.ids[idx]))

            # if label_data.ndim == 2:
            #     label_data = np.expand_dims(label_data, 0)

            out_data["label"]=label_data
            return out_data

        return out_data
