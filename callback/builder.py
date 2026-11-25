import importlib
import copy
from typing import Tuple,List

from pytorch_lightning.callbacks import *
from callback.normal_predict_callback import *
from callback.freezelayer_callback import *
from callback.fusion_predict_callback import *
from callback.fusion_multi_topk_predict_callback import *



# def generate_callback(cfg):
#     _callback_cfg=copy.deepcopy(cfg)
#     callbacks = []
#     if not isinstance(_callback_cfg,(Tuple,List)):
#          _callback_cfg=[_callback_cfg]
#     module_names = ['lightning.pytorch.callbacks','.freezelayer_callback','.normal_predict_callback']
#     modules = {}
#     for name in module_names:
#         modules[name] = importlib.import_module(name, __package__)

#     for cfg in _callback_cfg:
#         callback_name=cfg.pop('type')
#         params = cfg.pop('params',{})
#         for name in module_names:
#             callback_class = getattr(modules[name], callback_name, None)
#             if callback_class is not None:
#                 callback_class=callback_class(**params)
#                 callbacks.append(callback_class)
#                 break
#     return callbacks

def generate_callback(cfg):
    _callback_cfg=copy.deepcopy(cfg)
    callbacks = []
    if not isinstance(_callback_cfg,(Tuple,List)):
         _callback_cfg=[_callback_cfg]
    for cfg in _callback_cfg:
        callback_name=cfg.pop('type')
        params = cfg.pop('params',{})
        callback_class = getattr(importlib.import_module(__name__), callback_name, None)
        if callback_class is not None:
            callback_class = callback_class(**params)
            callbacks.append(callback_class)
    return callbacks
