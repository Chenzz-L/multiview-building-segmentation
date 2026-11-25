#!/conda/env python
# -*- coding:utf-8 -*-
import os
import argparse
import sys
from types import SimpleNamespace
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.cli import LightningCLI 
from data.datasets.dataset_RGB_DSM_COCO_Simple import Dataset_RGB_DSM_COCO_Simple
from models.normal_model_interface import NormalMInterface
from models.normal_model_interface_dinov3 import NormalMInterface_DINOv3
from models.normal_model_interface_dinov3_mask2former import NormalMInterface_DINOv3_mask2former
from models.fusion_model_interface_dinov3 import FusionMInterface_DINOv3
from data import build_Dataset,build_DataLoader
from yacs_config.default_config import *
from loggers import generate_logger
def main():
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('-cfg','--cfg_name', type=str,default="predict_Hangzhou_RGB_dinov3+upernet")
    cfg_args = parser.parse_args()

    cfg = load_yaml('./cfgs/' + cfg_args.cfg_name + '.yaml')
    model_key = ('model','callback','predict','metric')
    model_cfg = dict(filter(lambda item: item[0] in model_key, cfg.items()))
    model = NormalMInterface_DINOv3(**model_cfg)
    # model = NormalMInterface.load_from_checkpoint("/path/to/project/experiment\\test2\\train\\checkpoints\\epoch=34-"
    #                                               "val_loss=0.06-val_BinaryF1Score_epoch=0.90-other_metric=0.00.ckpt")

    # generate logger

    # logger= generate_logger(cfg['logger']) if 'logger' in cfg.keys() else None

    trainer = Trainer(**cfg['trainer'],profiler="simple")
    # if 'PREDICT_DATASET_CFG' in cfg:
    #     cfg['PREDICT_DATASET_CFG']['collate_fn'] = Dataset_RGB_DSM_COCO_Simple.collate_fn
    # build dataset and loader
    predict_set = build_Dataset(SimpleNamespace(**cfg['PREDICT_DATASET_CFG']))
    predict_loader, _ = build_DataLoader(predict_set, SimpleNamespace(**cfg['PREDICT_DATALOADER_CFG']))

    trainer.predict(model, predict_loader,return_predictions=False, ckpt_path="/path/to/nas/mvmapper_datamake/0_Hangzhou_v2/experiment/Hangzhou_XYZ_DSM_6_7_17_30_spatial_in_channels6/train/checkpoints/epoch=40-val_loss=0.0493-val_BinaryF1Score_epoch=0.8885-other_metric=0.000.ckpt")

    # cli=LightningCLI(MInterface,save_config_overwrite=True)

if __name__ == "__main__":
    main()
