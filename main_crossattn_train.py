#!/conda/env python
# -*- coding:utf-8 -*-
"""
训练入口: Cross-Attention Fusion (per-camera features + TDOM)
使用 cfg: cfg_Hangzhou_Fusion_dinov3+upernet_crossattn.yaml
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import argparse
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from lightning.pytorch.trainer import Trainer
from lightning.pytorch import seed_everything

from models.fusion_model_interface_dinov3_crossattn import FusionMInterface_DINOv3_CrossAttn

from data import build_Dataset, build_DataLoader
from yacs_config.default_config import *
from loggers import generate_logger


def main():
    # 强制固定所有随机种子 (Python, NumPy, PyTorch, CUDA, Dataloader Workers)
    seed_everything(46, workers=True)
    
    parser = argparse.ArgumentParser(description='Training Parameters (CrossAttn)')
    parser.add_argument(
        '-cfg', '--cfg_name', type=str,
        default="cfg_Hangzhou_Fusion_dinov3+upernet_sail_memory_attnr_taillocal_rawweighted_q16"
    )
    cfg_args = parser.parse_args()

    cfg = load_yaml('./cfgs/' + cfg_args.cfg_name + '.yaml')
    model_key = (
        'model',
        'optimizer',
        'lr_scheduler',
        'loss',
        'metric',
        'callback',
        'logger',
        'background_consistency',
        'view_supervision',
        'cross_attn_debug',
    )
    model_cfg = dict(filter(lambda item: item[0] in model_key, cfg.items()))
    model = FusionMInterface_DINOv3_CrossAttn(**model_cfg)

    # build dataset and loader
    train_set = build_Dataset(SimpleNamespace(**cfg['TRAIN_DATASET_CFG']))
    train_loader, train_sampler = build_DataLoader(train_set, SimpleNamespace(**cfg['TRAIN_DATALOADER_CFG']))

    val_set = build_Dataset(SimpleNamespace(**cfg['VAL_DATASET_CFG']))
    val_loader, val_sampler = build_DataLoader(val_set, SimpleNamespace(**cfg['VAL_DATALOADER_CFG']))

    logger = generate_logger(cfg['logger']) if 'logger' in cfg.keys() else None
    trainer = Trainer(**cfg['trainer'], logger=logger)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
