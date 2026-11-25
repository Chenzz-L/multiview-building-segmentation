import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib
import copy
from typing import Tuple,List

from torchmetrics import *
from torchmetrics import MetricCollection

from metric.Iou import *
from metric.boundary_metric import *

def generate_metric(metric_cfg):
    _metric_cfg=copy.deepcopy(metric_cfg)
    metrics = []
    if not isinstance(_metric_cfg,(Tuple,List)):
         _metric_cfg=[_metric_cfg]

    for metric in _metric_cfg:
        metric_name=metric.pop('type')
        params = metric.pop('params',{})
        metric_class = getattr(importlib.import_module(__name__), metric_name, None)
        if metric_class is not None:
            metric_class=metric_class(**params)
            metrics.append(metric_class)
    return MetricCollection(metrics)

# def generate_metric(metric_cfg):
#     _metric_cfg=copy.deepcopy(metric_cfg)
#     metrics = []
#     if not isinstance(_metric_cfg,(Tuple,List)):
#          _metric_cfg=[_metric_cfg]
#     module_names = ['torchmetrics','.Iou']
#     modules = {}
#     for name in module_names:
#         modules[name] = importlib.import_module(name, __package__)

#     for metric in _metric_cfg:
#         metric_name=metric.pop('type')
#         params = metric.pop('params',{})
#         # metric_class=getattr(importlib.import_module('torchmetrics'),metric_name)(**params)
#         for name in module_names:
#             metric_class=getattr(modules[name],metric_name,None)
#             if metric_class is not None:
#                 metric_class=metric_class(**params)
#                 metrics.append(metric_class)
#                 break
#     return MetricCollection(metrics)

# def instancialize(self, Model, **other_args):
#         class_args = inspect.getargspec(Model.__init__).args[1:]
#         inkeys = self.hparams.keys()
#         args1 = {}
#         for arg in class_args:
#             if arg in inkeys:
#                 args1[arg] = getattr(self.hparams, arg)
#         args1.update(other_args)
#         return Model(**args1)
