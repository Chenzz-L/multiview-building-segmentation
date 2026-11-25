import os

import numpy as np

from data.inference_datasets.inference_dataset_Base import InferenceBaseDataset
from data.reading_data_tool import *
from utils.raster_io import *
from data.datasets.dataste_support import *

@regist_dataset()
class InferenceDataset_DSM(InferenceBaseDataset):
    def __init__(self,
                 data_dir: str,
                 dsm_none_value: float = -32767,
                 mask=False,
                 standardization: bool = True,
                 img_type:str='tif',
                 indices_name=None,
                 multi_branch_for_dsm=False,
                 multi_branch_channels:int=3):
        assert os.path.exists(data_dir)
        self.data_dir = data_dir
        self._dsm_dir = os.path.join(data_dir, 'dsm')
        self.dsm_none_value=dsm_none_value
        self.mask=mask
        self.standardization = standardization
        self.multi_branch_for_dsm=multi_branch_for_dsm
        self.multi_branch_channels=multi_branch_channels
        super(InferenceDataset_DSM, self).__init__(data_dir,standardization,img_type,indices_name)

    def set_dir(self, image_dir_name: str, label_dir_name: str, dsm_dir_name: str = None):
        self._image_dir = os.path.join(self.data_dir, image_dir_name)
        self._label_dir = os.path.join(self.data_dir, label_dir_name)
        self._dsm_dir = os.path.join(self.data_dir, dsm_dir_name)
        assert os.path.isdir(self.data_dir)
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)
        assert os.path.isdir(self._dsm_dir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.standardization:
            if self.stat_data is None :
                raise Exception("statics data has None value")
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)
        assert os.path.isdir(self._dsm_dir)

        image_name=self.ids[idx]

        # read image
        image_data=self.open(os.path.join(self._image_dir, self.ids[idx]),
                            channel_sum=3,
                            standardization=self.standardization,mean=0,std=255.0)
        
        # read dsm
        dsm_data=self.open(os.path.join(self._dsm_dir, '%s' % self.ids[idx]),
                            standardization=self.standardization,mean=0,std=self.stat_data[-1][-2],
                            mask=self.mask,mask_value=self.dsm_none_value,none_value=self.dsm_none_value)
            
        if self.multi_branch_for_dsm:
            dsm_data=np.tile(dsm_data,[self.multi_branch_channels,1,1])
        image_data = np.concatenate((image_data, dsm_data), axis=0)

        out_data = {"image": image_data, "name": image_name}
        # read label
        if self._label_dir is not None:

            label_data=self.open(os.path.join(self._label_dir, '%s' % self.ids[idx]))
            # if label_data.ndim == 2:
            #     label_data = np.expand_dims(label_data, 0)

            out_data["label"]=label_data
            return out_data

        return out_data
