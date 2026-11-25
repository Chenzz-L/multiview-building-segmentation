import os
import glob

import numpy as np
from data.inference_datasets.inference_dataset_Base import InferenceBaseDataset
from data.reading_data_tool import *
from utils.raster_io import *
from data.datasets.dataste_support import *
import pandas as pd

__all__ = ['InferenceDataset_Hangzhou_RGB']

@regist_dataset()
class InferenceDataset_Hangzhou_RGB(InferenceBaseDataset):
    def __init__(self,
                 data_dir: str,
                 indices_name:str):
        assert os.path.exists(data_dir)
        self.data_dir = data_dir
        self._image_dir = os.path.join(data_dir, 'image')
        self._label_dir =os.path.join(data_dir, 'label') if os.path.exists(os.path.join(data_dir, 'label')) else None
        if indices_name!=None:
            self.indices = pd.read_csv(os.path.join(data_dir, f'{indices_name}.csv'))
        else:
            self.indices = pd.read_csv(os.path.join(data_dir, 'indices.csv'))
            
        self.ids = self.indices['name'].tolist()
        

    def set_dir(self, image_dir_name: str, label_dir_name: str, dsm_dir_name: str = None):
        self._image_dir = os.path.join(self.data_dir, image_dir_name)
        self._label_dir = os.path.join(self.data_dir, label_dir_name)
        assert os.path.isdir(self.data_dir)
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)

    def __getitem__(self, idx):
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)

        image_name=self.ids[idx]

        image_data=self.open(os.path.join(self._image_dir, '%s' % self.ids[idx]),channel_sum=3)
        # read label
        label_data=self.open(os.path.join(self._label_dir, '%s' % self.ids[idx])).astype(np.int64)
        # read dsm

        out_data = {"image": image_data, "name": image_name, 'label':label_data}
        return out_data


# if __name__=='__main__':
#     from utils.raster_io import fastly_save_image
#     import gdal

#     data_pipeline=[{'type' : 'RandomResizedCrop','height': 384,'width' : 384,'scale' : [0.08,1.0],
#                     'ratio' : [0.75,1.3333333333333333],'interpolation' : 2,'always_apply':True},
#                    {'type': 'Flip' , 'p' : 0.5},
#                    {'type' : 'RandomBrightnessContrast' , 'p' : 0.5},
#                    {'type' : 'ColorJitter'},
#                    {'type' : 'ElasticTransform'},
#                    {'type' : 'GridDistortion'}]
#     dataset = Dataset_Melbourne_DSM(r"/path/to/user/NAS/gwx/DeepL/Melbourne/IMG_MV/samples",
#                           False,-99999,False,None,data_pipeline,indices_name='train_indices')

#     data_loader = tdata.DataLoader(dataset=dataset, batch_size=1,
#                                    num_workers=0, shuffle=False,  # cfg.shuffle,
#                                    pin_memory=True)

#     batch_iterator = iter(data_loader)
#     for batch in batch_iterator:
#         x, y= batch
#         fastly_save_image(x.squeeze(0).numpy(),'/path/to/user/NAS/gwx/DeepL/Melbourne/image.tif',gdal_type=gdal.GDT_Float32)
#         fastly_save_image(y.squeeze(0).numpy(),'/path/to/user/NAS/gwx/DeepL/Melbourne/label.tif',gdal_type=gdal.GDT_Float32)
#         break
#         if torch.isnan(x).any():
#             print('found image nan')
#         if torch.isinf(x).any():
#             print('found image inf')

#         if torch.isnan(y).any():
#             print('found label nan')
#         if torch.isinf(y).any():
#             print('found label inf')
#         for i in range(x.shape[0]):
#             out = x[i].numpy()
#             fastly_save_image(out, os.path.join(r"/path/to/user/NAS/gwx/experiment_result/", 'img'+str(i) + ".tif"),
#                               gdal_type=gdal.GDT_Float32)
#             fastly_save_image(y[i].numpy(),
#                               os.path.join(r"/path/to/user/NAS/gwx/experiment_result/", 'label'+str(i) + ".tif"),
#                               gdal_type=gdal.GDT_Float32)

#         print("ok")

#         break
