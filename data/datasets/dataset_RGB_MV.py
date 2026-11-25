# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np

from data.datasets.data_utils import expand_data_2_BCHW,expand_datalist_2_BCHW
from data.datasets.dataset_Base import BaseDataset
from data.transforms.augmentation import *
from data.datasets.dataste_support import *
from utils.raster_io import RasterIO,fastly_save_image
from utils.make_samples import *

# from utils.basic_func import *
# from utils.metrics import *

__all__ = ['Dataset_RGB_Fusion']

@regist_dataset()
class Dataset_RGB_Fusion(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 grid_dir: str,
                 xyz_label_dir: str,
                 xyz_img_pair_file: str,
                 grid_scale: float,
                 augmentation: bool = False,
                 standardization: bool = True,
                 clip: list = None,
                 data_pipeline=None):
        super().__init__(data_dir, augmentation, standardization, None,None)
        # self.pair_file=pair_file
        self.views_meta = self.build_pair()
        self.grid_dir = grid_dir
        self.xyz_label_dir = xyz_label_dir
        self.xyz_img_dict = self.read_xyz_img_pair_file(xyz_img_pair_file)
        self.grid_scale = grid_scale
        # self.curent_bacth_index=None
        if clip is not None:
            assert len(clip) == 2
            self.views_meta = self.views_meta[clip[0]:clip[1]]

        if augmentation and data_pipeline:
            if not 'ToFloat' in data_pipeline[0].values():# 自定义ToFloat 
                data_pipeline.insert(0,{'type':'ToFloat','max_value':255})
            self.data_pipeline=build_agument_pipeline(data_pipeline,True)
            self.augmentation=augmentation
            
    # def get_current_bacthinfo(self):
    #     return self.views_meta[self.curent_bacth_index]
    
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
            if self.stat_data is None:
                raise Exception("statics data has None value")
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)
        imgs = []
        label_datas = []
        grid_datas = []
        # self.curent_bacth_index=idx
        
        # read xyz data
        model_predict_name = os.path.splitext(os.path.basename(self.views_meta[idx][0]))[0]
        xyz_num = model_predict_name.split("_")[-3]
        num = model_predict_name.split("_")[-2:]
        num = "_".join(num)
        xyz_label_file = os.path.join(self.xyz_label_dir, self.xyz_img_dict[xyz_num] + "_" + num + ".tif")
        xyz_label_data=self.open(xyz_label_file)

        # read image
        for j, file in enumerate(self.views_meta[idx]):
            # read image
            # image_data=self.open(os.path.join(self._image_dir, '%s' % file),
            #                 channel_sum=3,
            #                 standardization=self.standardization,mean=0,std=255.0)
            image_data=self.open(os.path.join(self._image_dir, '%s' % file),
                            channel_sum=3,
                            standardization=not self.augmentation,mean=0,std=255.0)
            # read label
            label_data=self.open(os.path.join(self._label_dir, '%s' % file))

            # read grid
            grid_data=self.open(os.path.join(self.grid_dir, file),scale=self.grid_scale)
            
            grid_datas.append(grid_data)
            imgs.append(image_data)
            label_datas.append(label_data)

        # if self.augmentation:
        #     xyz_label_data,grid_datas,img_labels=data_augumentation_group_views(xyz_label_data,grid_datas,imgs,label_datas)
        #     imgs, label_datas=img_labels
        if self.augmentation and self.data_pipeline!=None:
            for i,img in enumerate(imgs):
                imgs[i]=seg_augmentation_imageonly(img,self.data_pipeline)

        expand_datalist_2_BCHW(imgs)
        expand_datalist_2_BCHW(label_datas)
        expand_datalist_2_BCHW(grid_datas)
        xyz_label_data = expand_data_2_BCHW(xyz_label_data)

        img=np.ascontiguousarray(np.concatenate(imgs,axis=0))
        grid_data = np.ascontiguousarray(np.concatenate(grid_datas,axis=0).transpose((0, 2, 3, 1)))
        label_data = np.ascontiguousarray(np.concatenate(label_datas, axis=0))
        return img, grid_data, label_data, xyz_label_data



if __name__=='__main__':
    import torch
    import torch.utils.data as tdata
    from data.datasets.data_utils import fusion_collate
    import gdal
    from utils.raster_io import fastly_save_image
    data_pipeline=[{'type' : 'RandomBrightnessContrast' , 'p' : 0.5},
                   {'type': 'Flip' , 'p' : 0.5},
                   {'type' : 'ColorJitter'}]
    dataset=Dataset_RGB_Fusion(r"/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none",
                                r"/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none/grid_none",
                                r"/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/train/samples_none/label",
                                r"/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/XYZ_xy_name_map.txt",
                                0.25,False, False,data_pipeline=None)
    data_loader = tdata.DataLoader(dataset=dataset, batch_size=1,
                                   num_workers=4, shuffle=True,
                                   pin_memory=False, collate_fn=fusion_collate)
    batch_iterator=iter(data_loader)
    n=0
    for batch in batch_iterator:
        x,grid,image_label,xyz_label,batch_info=batch
        if x.shape[0]==5:
            if not os.path.exists(f'/path/to/user/NAS/gwx/DeepL/Xuzhou/mv_test/{n}'):
                os.makedirs(f'/path/to/user/NAS/gwx/DeepL/Xuzhou/mv_test/{n}')
            for i in range(5):
                fastly_save_image(x[i].squeeze(0).numpy()*255,f'/path/to/user/NAS/gwx/DeepL/Xuzhou/mv_test/{n}/{i}.tif',gdal_type=gdal.GDT_Byte)
            n+=1
            
        # if torch.isnan(x).any():
        #     print('found nan')
        # if (x==0).all():
        #     print("empty")
    
    # model = hrnet_mv32(in_channels=3, out_channels=1)
    # load_pretrain_with_checkpoint(model,r"/path/to/user/NAS/gwx/experiment_result/experiment_stage2_send_loss_iou_v2_hrnet_mv32_Roof_DOM_MV_Dataset_IMG_MV/weights/hrnet_mv32_epoch042.pth")
    # model.cuda()
    # model.eval()
    # # model.train()
    # batch_iterator=iter(data_loader)
    # global_stats = np.zeros(shape=(2,3))
    # with torch.no_grad():
    #     for batch in batch_iterator:
    #         x,grid,y1,y2,batch_info=batch
    #         x = x.cuda()
    #         grid = grid.cuda()
    #         batch_info = batch_info.cuda()
    #         y1 = y1.cuda()
    #         y2 = y2.cuda()
    #         gen_y = model(x, grid, batch_info)
    #         pred_y = binarize(gen_y[0])
    #         tp, fp, fn = calc_tp_fp_fn_from_tensor(pred_y, y1)
    #         global_stats[0]+=[tp,fp,fn]
    #
    #         pred_y = binarize(gen_y[1])
    #         tp, fp, fn = calc_tp_fp_fn_from_tensor(pred_y, y2)
    #         global_stats[1]+=[tp,fp,fn]
    #         del x
    #         del grid
    #         del y1
    #         del y2
    #         del batch_info
    #         torch.cuda.empty_cache()
    #
    # train_iou, train_precision, train_recall, train_f1score = calculate_acc_from_tp_tn_fp(global_stats[0][0], global_stats[0][1], global_stats[0][2])
    # print(train_iou)
    # train_iou, train_precision, train_recall, train_f1score = calculate_acc_from_tp_tn_fp(global_stats[1][0], global_stats[1][1], global_stats[1][2])
    # print(train_iou)

    # dataset = Roof_DOM_MV_Dataset(r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\samples_none",
    #                               r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\samples_none\grid_none",
    #                               r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\samples_none\label",
    #                               r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\XYZ_xy_name_map.txt",
    #                               1, False, False)
    # data_loader = tdata.DataLoader(dataset=dataset, batch_size=1,
    #                                num_workers=1, shuffle=True,
    #                                pin_memory=False, collate_fn=dataset_collate)
    # for data in data_loader:
    #     x,grid,y1,y2,batch_info=data
    #     for i in range(x.shape[0]):
    #       out = x[i].numpy()
    #       fastly_save_image(out,os.path.join(r"/path/to/project/Users\35416\Desktop\check_data\dataset\image",str(i)+".tif"),gdal_type=gdal.GDT_Float32)
    #       t_grid=grid[i].numpy()
    #       t_grid=np.transpose(t_grid,[2,0,1])
    #       fastly_save_image(t_grid,
    #                         os.path.join(r"/path/to/project/Users\35416\Desktop\check_data\dataset\grid", str(i) + ".tif"),
    #                         gdal_type=gdal.GDT_Float32)
    #       fastly_save_image(y1[i].numpy(),
    #                         os.path.join(r"/path/to/project/Users\35416\Desktop\check_data\dataset\label1",str(i)+".tif"),
    #                         gdal_type=gdal.GDT_Float32)

    #     for i in range(y2.shape[0]):
    #         fastly_save_image(y2[i].numpy(),
    #                           os.path.join(r"/path/to/project/Users\35416\Desktop\check_data\dataset\label2", str(i) + ".tif"),
    #                           gdal_type=gdal.GDT_Float32)
    #     break
