#!/conda/env python
# -*- coding:utf-8 -*-
import os
# os.environ["OMP_NUM_THREADS"] = "1" 
# os.environ["MKL_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 2. 强制禁用 OpenCV 多线程 (防止死锁)
import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
# import torch.multiprocessing as mp
import argparse
import sys
from types import SimpleNamespace
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.cli import LightningCLI

from models.model_interface import MInterface
from models.normal_model_interface import NormalMInterface
from models.normal_model_interface_dinov3 import NormalMInterface_DINOv3
from models.normal_model_interface_dinov3_mask2former import NormalMInterface_DINOv3_mask2former
from models.normal_model_interface_dinov3_solov2 import NormalMInterface_DINOv3_SOLOv2
from models.fusion_model_interface_dinov3 import FusionMInterface_DINOv3

from data import build_Dataset,build_DataLoader
from yacs_config.default_config import *
from loggers import generate_logger 
from data.datasets.dataset_RGB_DSM_COCO_Simple import Dataset_RGB_DSM_COCO_Simple
from data.datasets.dataset_RGB_DSM_COCO_Simple_solov2 import Dataset_RGB_DSM_COCO_Simple_solov2

from callback.resample_callback import ResampleDatasetCallback
from callback.builder import generate_callback
# from osgeo import gdal
# gdal.UseExceptions()


def main():
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('-cfg','--cfg_name', type=str,default="cfg_Hangzhou_XYZ_RGB_dinov3+upernet")
    cfg_args = parser.parse_args()

    cfg=load_yaml('./cfgs/' + cfg_args.cfg_name + '.yaml')
    model_key=('model','optimizer','lr_scheduler','loss','metric','callback','logger')
    model_cfg=dict(filter(lambda item: item[0] in model_key, cfg.items()))
    model=NormalMInterface_DINOv3(**model_cfg)             
    # print(model.model.encoder.backbone.blocks[0].attn)
    # generate logger
    # if 'TRAIN_DATALOADER_CFG' in cfg:
    #     cfg['TRAIN_DATALOADER_CFG']['collate_fn'] = Dataset_RGB_DSM_COCO_Simple.collate_fn
    
    # if 'VAL_DATALOADER_CFG' in cfg:
    #     cfg['VAL_DATALOADER_CFG']['collate_fn'] = Dataset_RGB_DSM_COCO_Simple.collate_fn
    # build dataset and loader  
    train_set = build_Dataset(SimpleNamespace(**cfg['TRAIN_DATASET_CFG']))
    train_loader, train_sampler = build_DataLoader(train_set, SimpleNamespace(**cfg['TRAIN_DATALOADER_CFG']))

    # build dataset and loader
    val_set = build_Dataset(SimpleNamespace(**cfg['VAL_DATASET_CFG']))
    val_loader, val_sampler = build_DataLoader(val_set, SimpleNamespace(**cfg['VAL_DATALOADER_CFG']))
    
    # 准备 Callbacks 列表
    # -----------------------------------------------------------
    # all_callbacks = []

    # if 'callbacks' in cfg['trainer']:
    #     yaml_callbacks_cfg = cfg['trainer'].pop('callbacks') 
    #     generated_list = generate_callback(yaml_callbacks_cfg)
    #     all_callbacks.extend(generated_list)
    
    # resample_cb = ResampleDatasetCallback(train_set)
    # all_callbacks.append(resample_cb)
    # cfg['trainer']['reload_dataloaders_every_n_epochs'] = 1
    
    logger= generate_logger(cfg['logger']) if 'logger' in cfg.keys() else None
    # trainer=Trainer(**cfg['trainer'],callbacks=all_callbacks,logger=logger)
    trainer=Trainer(**cfg['trainer'],logger=logger)
    
    trainer.fit(model, train_loader, val_loader)

    # cli=LightningCLI(MInterface,save_config_overwrite=True)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
