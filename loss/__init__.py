#!/conda/env python
# -*- coding:utf-8 -*-
from typing import Tuple,List
import copy
import torch.nn as nn
import torch

from loss.losses import *

LOSS_FUNC={
    'focalloss':FocalLoss,
    'labelsmoothsoftmaxce':LabelSmoothSoftmaxCE,
    'sigmoiddiceloss':SigmoidDiceLoss,
    'spatialdiceloss':SpatialDiceLoss,
    'lovaszsoftmaxloss':LovaszSoftmaxLoss,
    'bcewithlogitsloss':BCEWithLogitsLoss
}


def build_loss(cfg):
    loss_func_list=copy.deepcopy(cfg)
    loss_func = []
    if not isinstance(loss_func_list,(Tuple,List)):
        loss_func_list=[loss_func_list]

    for loss in loss_func_list:
        loss_name=loss.pop('type').lower()
        params = loss.pop('params',{})
        if loss_name in LOSS_FUNC.keys():
            loss_func.append(LOSS_FUNC[loss_name](**params))
        # elif loss_name=='CrossEntropyLoss':
        #     loss_func.append(nn.CrossEntropyLoss(torch.tensor(cfg.sample_weights).cuda(),reduction=cfg.reduction))
        elif hasattr(nn, loss):
            loss_func.append(getattr(nn,loss)(**params))
        else:
            raise ValueError("loss is not exist")
    return tuple([loss_func],)

def build_loss_from_str(loss_func_list):
    loss_func = []
    if not isinstance(loss_func_list,(Tuple,List)):
        loss_func_list=[loss_func_list]

    for loss in loss_func_list:
        if loss.lower() in LOSS_FUNC.keys():
            loss_func.append(LOSS_FUNC[loss.lower()])
        elif loss.lower()=='CrossEntropyLoss':
            loss_func.append(nn.CrossEntropyLoss())
        elif hasattr(nn, loss):
            loss_func.append(getattr(nn,loss)())
        else:
            raise ValueError("loss is not exist")

    return tuple(loss_func)
