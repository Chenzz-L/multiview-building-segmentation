# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import pandas as pd

from data.datasets.dataset_Base import BaseDataset
from data.datasets.data_utils import expand_data_2_BCHW,expand_datalist_2_BCHW,get_xyzname_from_pairfile
from data.transforms.augmentation import *
from data.datasets.dataste_support import *
from utils.make_samples import *



__all__ = ['Dataset_Fusion_image_xyz_padding']

@regist_dataset()
class Dataset_Fusion_image_xyz_padding(BaseDataset):
    def __init__(self,
                 img_data_dir: str,
                 grid_dir: str,
                 xyz_data_dir: str,
                 xyz_img_pair_file: str,
                 grid_scale:float,
                 img_input_name_channels_dcit: dict = {'image':3,'dsm':1},
                 xyz_input_name_channels_dcit: dict = {'image':3,'dsm':1},
                 augmentation: bool = False,
                 clip: list = None,
                 data_pipeline=None,
                 indices_name='indices'):
        assert os.path.exists(img_data_dir)
        assert os.path.exists(xyz_data_dir)
        assert os.path.exists(grid_dir)
        self.img_data_dir = img_data_dir
        self.xyz_data_dir=xyz_data_dir
        self.gird_dir = grid_dir
        self._img_input_name_channels_dcit = img_input_name_channels_dcit
        self._xyz_input_name_channels_dcit = xyz_input_name_channels_dcit

        self.augmentation = augmentation
        # get indices, return pd.DataFrame, a python dictionary
        self.indices = pd.read_csv(os.path.join(img_data_dir, indices_name+'.csv'))
        self.ids = self.indices['name'].tolist()
        self.views_meta = self.build_pair()

        self.xyz_img_dict = self.read_xyz_img_pair_file(xyz_img_pair_file)
        self.grid_scale = grid_scale
        if clip is not None:
            assert len(clip) == 2
            self.views_meta = self.views_meta[clip[0]:clip[1]]

        if augmentation and data_pipeline:
            if not 'ToFloat' in data_pipeline[0].values():# 自定义ToFloat
                data_pipeline.insert(0,{'type':'ToFloat','max_value':1})
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
        xyz_inputs = []
        xyz_suffix=get_xyzname_from_pairfile(self.views_meta[idx][0],self.xyz_img_dict)
        for key,value in self._xyz_input_name_channels_dcit.items():
            file = os.path.join(self.xyz_data_dir, key, xyz_suffix)
            xyz_image_data=self.open(file,channel_sum=value)
            xyz_inputs.append(xyz_image_data)
        xyz_image_data = np.concatenate(xyz_inputs, axis=0)
        xyz_label_data=self.open(os.path.join(self.xyz_data_dir,'label',xyz_suffix))
        
        
        for j, file in enumerate(self.views_meta[idx]):
            img_inputs=[]
            for key,value in self._img_input_name_channels_dcit.items():
                file_path = os.path.join(self.img_data_dir, key, file)
                image_data=self.open(file_path,channel_sum=value)
                
                if self.augmentation and self.data_pipeline!=None and key=='image':
                    image_data=seg_augmentation_imageonly(image_data,self.data_pipeline)
                img_inputs.append(image_data)
            image_data = np.concatenate(img_inputs, axis=0)
            imgs.append(image_data)
            
            # read label
            label_data=self.open(os.path.join(self.img_data_dir,'label', '%s' % file))
            label_datas.append(label_data)
                
            # read grid
            grid_data=self.open(os.path.join(self.gird_dir, file),scale=self.grid_scale)
            grid_datas.append(grid_data)

        expand_datalist_2_BCHW(imgs)
        expand_datalist_2_BCHW(label_datas)
        expand_datalist_2_BCHW(grid_datas)
        xyz_label_data = expand_data_2_BCHW(xyz_label_data)
        xyz_image_data = expand_data_2_BCHW(xyz_image_data)
        

        # img=np.concatenate(imgs,axis=0)
        grid_data = np.concatenate(grid_datas,axis=0)
        # label_data = np.concatenate(label_datas, axis=0)
        grid_data = grid_data.transpose((0, 2, 3, 1))
        return imgs,xyz_image_data, grid_data, label_data, xyz_label_data


# if __name__=='__main__':

#     import pandas as pd
#     from data.datasets.data_utils import dataset_collate,dataset_collate_padding
#     import torch.utils.data as tdata 
#     import gdal
#     from utils.raster_io import fastly_save_image
#     data_pipeline=[{'type' : 'RandomBrightnessContrast' , 'p' : 0.5},
#                    {'type': 'Flip' , 'p' : 0.5},
#                    {'type' : 'ColorJitter'}]
#     dataset=Dataset_Fusion_image_xyz_padding("/path/to/user/NAS/gwx/DeepL/Melbourne/IMG_MV/samples",
#                                 "/path/to/user/NAS/gwx/DeepL/Melbourne/IMG_MV/samples/grid_none",
#                                 "/path/to/user/NAS/gwx/DeepL/Melbourne/XYZ/samples",
#                                 "/path/to/user/NAS/gwx/DeepL/Melbourne/XYZ_xy_name_map.txt",
#                                 1,augmentation=False,data_pipeline=data_pipeline,
#                                 xyz_input_name_channels_dcit={'dsm':1},img_input_name_channels_dcit={'image':1,'dsm':1},)
#                                 # ,indices_name='singleview_indices')

#     data_loader = tdata.DataLoader(dataset=dataset, batch_size=2,
#                                    num_workers=0, shuffle=False,  # cfg.shuffle,
#                                    pin_memory=True, collate_fn=dataset_collate_padding)
    
#     import torch.nn.functional as F
#     import torch
#     batch_iterator=iter(data_loader)
#     for batch in batch_iterator:
#         img_x,xyz_x,grid,image_label,y2,batch_info=batch
#         x=torch.tensor(img_x[0].unsqueeze(0),dtype=torch.float32)*255
#         y=torch.tensor(grid[0].unsqueeze(0))
#         out = F.grid_sample(x,y , align_corners=True)
#         out = out.numpy().squeeze(axis=0)

#         out_file=r"/path/to/user/NAS/gwx/DeepL/Melbourne/trans3.tif"
#         ref_file=r"/path/to/user/NAS/gwx/DeepL/Melbourne/XYZ/samples/dsm/XYZ_5_1350_0150.tif"
#         fastly_save_image(out, out_file, ref_file=ref_file,gdal_type=gdal.GDT_Float32)
#         break

        # fastly_save_image(img_x[0].squeeze(0).numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/image.tif',gdal_type=gdal.GDT_Float32)
        # fastly_save_image(xyz_x[0].squeeze(0).numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/image22.tif',gdal_type=gdal.GDT_Float32)
        # fastly_save_image(image_label[0].squeeze(0).numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/label.tif',gdal_type=gdal.GDT_Float32)
        # break
        
        # if len(img_x)==3:# and np.sum(image_label[0].squeeze(0).numpy())/(512*512)>0.4:
        #     for i in range(img_x.shape[0]):
        #         fastly_save_image(img_x[i,0:3].numpy()*255,'/path/to/user/NAS/gwx/code/seg_framer_v2/image'+str(i)+'.tif',gdal_type=gdal.GDT_Float32)
        #         fastly_save_image(img_x[i,3:].numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/image_dsm'+str(i)+'.tif',gdal_type=gdal.GDT_Float32)
        #         fastly_save_image(image_label[i].numpy()*255,'/path/to/user/NAS/gwx/code/seg_framer_v2/label'+str(i)+'.tif')
        #         fastly_save_image(np.transpose(grid[i].numpy(),[2,0,1]),
        #                             '/path/to/user/NAS/gwx/code/seg_framer_v2/grid'+str(i)+'.tif',gdal_type=gdal.GDT_Float32)

        #     for i in range(y2.shape[0]):
        #         fastly_save_image(y2[i].numpy()*255,'/path/to/user/NAS/gwx/code/seg_framer_v2/xyz_label'+str(i)+'.tif')
        #         fastly_save_image(xyz_x[i,0:3].numpy()*255,'/path/to/user/NAS/gwx/code/seg_framer_v2/xyz'+str(i)+'.tif',gdal_type=gdal.GDT_Float32)
        #         fastly_save_image(xyz_x[i,3:].numpy(),'/path/to/user/NAS/gwx/code/seg_framer_v2/xyz_dsm'+str(i)+'.tif',gdal_type=gdal.GDT_Float32)
        #     break

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
