import importlib
import copy
from typing import Tuple,List

def generate_logger(cfg):
    _loggers_cfg=copy.deepcopy(cfg)
    loggers = []
    if not isinstance(_loggers_cfg,(Tuple,List)):
         _loggers_cfg=[_loggers_cfg]

    for cfg in _loggers_cfg:
        logger_name=cfg.pop('type')
        params = cfg.pop('params',{})
        logger=getattr(importlib.import_module('lightning.pytorch.loggers'),logger_name)(**params)
        loggers.append(logger)
    return loggers
