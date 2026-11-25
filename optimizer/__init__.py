#!/conda/env python
# -*- coding:utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import inspect
from typing import List,Tuple
from functools import partial
import importlib
from collections.abc import Mapping

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.distributed as dist
from torch.nn import ModuleList
import numpy as np

from models.hrnet_mv import *
from optimizer.lr_scheduler import *
from yacs_config.default_config import *
from cfgs.builder import BuildingConfigFromYaml

support_optimizer=dict(inspect.getmembers(optim, inspect.isclass))


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]
    except (KeyError, TypeError, AttributeError):
        return default


def get_init_param_from_optimizer(optimizer_name:str,except_param=("",)):
    default_except_param=['self',]
    default_except_param+=except_param
    default_except_param=list(set(default_except_param))
    
    params = inspect.signature(support_optimizer[optimizer_name].__dict__['__init__']).parameters
    out_param={}
    for name,param in params.items():
        if name in default_except_param:
            continue
        if not isinstance(param.default,inspect._empty):
            out_param[name]=param.default
        else:
            out_param[name]="no_default"

    return out_param

def init_optimizer_with_cfg(optimizer_name:str,cfg,except_param=("",)):
    if optimizer_name is None:
        raise KeyError(f"optimizer type is None, optimizer cfg = {cfg}")
    params=get_init_param_from_optimizer(optimizer_name,("params","lr"))

    for key ,val in params.items():
        key_param_cfg=_cfg_get(cfg, key, None)
        # if not (val=="no_default" and kay_param_cfg==None):
        #     raise "positional-only val must given"
        assert  not (val=="no_default" and key_param_cfg==None),"positional-only val must given"
        if key_param_cfg!=None:
            params[key]=key_param_cfg

    opt=partial(support_optimizer[optimizer_name],**params)
    return opt

def plot_lr(optimizer,scheduler, title='', labels=('base'), nrof_epoch=100):
    lr_li = [[] for _ in range(len(labels))]
    epoch_li = list(range(nrof_epoch))
    for epoch in epoch_li:
        optimizer.step()
        scheduler.step()  # 调用step()方法,计算和更新optimizer管理的参数基于当前epoch的学习率
        lr = scheduler.get_last_lr()  # 获取当前epoch的学习率
        # for i, data in enumerate(optimizer.param_groups):
        #     lr = data['lr']
        #     lr_li[i].append(lr)
        for i in range(len(labels)):
            lr_li[i].append(lr[i])
    for lr, label in zip(lr_li, labels):
        plt.plot(epoch_li, lr, label=label)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title(title)
    plt.legend()
    plt.show()

def build_optimizer(model, optim_cfg):
    node_num=dist.get_world_size() if dist.is_initialized() else 1
    cfg_lr=np.array(_cfg_get(optim_cfg, 'lr'))*node_num
    no_ddp_model=model
    model_name = no_ddp_model.__class__.__name__
    seg_encoder_decoder=['SegEncoderDecoderFusion','SegEncoderDecoder']
    parmas_layers_cfg = _cfg_get(optim_cfg, 'parmas_layers', None)
    if parmas_layers_cfg:
        if model_name in seg_encoder_decoder and (isinstance(no_ddp_model.backbone,ModuleList) or isinstance(no_ddp_model.decode_head,ModuleList)):
            module_generater=partial(no_ddp_model.named_modules)
        else:
            module_generater=partial(no_ddp_model.named_children)

        param_dict={}
        if  isinstance(parmas_layers_cfg[0],(Tuple,List)):
            lr = cfg_lr[-1]
            parmas_layers=parmas_layers_cfg
        else:# 只有一组参数
            lr = cfg_lr
            cfg_lr=[lr]
            parmas_layers=[parmas_layers_cfg]

        for name,module in module_generater():
            for i, layers in enumerate(parmas_layers):
                for layer_name in layers:
                    if layer_name == name:
                        if i in param_dict.keys():
                            param_dict[i]+=list(module.parameters())
                        else:
                            param_dict[i] = list(module.parameters())
        # 参数分为多组

        param_group=[{'params':filter(lambda p:p.requires_grad,value),'initial_lr': cfg_lr[key], 'lr': cfg_lr[key]}
                    for key,value in param_dict.items()]
    else:
        param_group = [{'params': filter(lambda p:p.requires_grad,model.parameters()), 'initial_lr': cfg_lr,'lr':cfg_lr}]
        lr = cfg_lr


    opt=init_optimizer_with_cfg(_cfg_get(optim_cfg, 'type'),optim_cfg)
    opt=opt(params=param_group, lr=lr)
    return opt


def build_scheduler(optimizer,scheduler_cfg):
    lr_scheduler=None
    if scheduler_cfg is not None:
        scheduler_cfg_=scheduler_cfg.copy()
        lr_scheduler_name=scheduler_cfg_.pop('type',False)
        lr_scheduler=getattr(scheduler,lr_scheduler_name)(optimizer,**scheduler_cfg_)

        # if optim_cfg.lr_scheduler.lower()=='multisteplr':
        #     lr_scheduler=scheduler.MultiStepLR(optimizer,milestones=optim_cfg.milestones,
        #                                     gamma=optim_cfg.gamma,
        #                                     last_epoch=optim_cfg.last_epoch)
        # elif optim_cfg.lr_scheduler.lower()=='reducelronplateau':
        #     mode=optim_cfg.mode if hasattr(optim_cfg, 'mode') and optim_cfg.mode is not None else 'min'
        #     factor=optim_cfg.factor if hasattr(optim_cfg, 'factor') and optim_cfg.factor is not None else 0.1
        #     patience=optim_cfg.patience if hasattr(optim_cfg, 'patience') and optim_cfg.patience is not None else 10
        #     verbose=optim_cfg.verbose if hasattr(optim_cfg, 'verbose') and optim_cfg.verbose is not None else False
        #     threshold=optim_cfg.threshold if hasattr(optim_cfg, 'threshold') and optim_cfg.threshold is not None else 0.0001
        #     threshold_mode=optim_cfg.threshold_mode if hasattr(optim_cfg, 'threshold_mode') and optim_cfg.threshold_mode is not None else 'rel'
        #     cooldown=optim_cfg.cooldown if hasattr(optim_cfg, 'cooldown') and optim_cfg.cooldown is not None else 0
        #     min_lr=optim_cfg.min_lr if hasattr(optim_cfg, 'min_lr') and optim_cfg.min_lr is not None else 0
        #     eps=optim_cfg.eps if hasattr(optim_cfg, 'eps') and optim_cfg.eps is not None else 1e-08
        #     lr_scheduler=scheduler.ReduceLROnPlateau(optimizer,mode=mode,factor=factor,patience=patience,
        #                                             verbose=verbose,threshold=threshold,threshold_mode=threshold_mode,
        #                                             cooldown=cooldown,min_lr=min_lr,eps=eps)
        # elif optim_cfg.lr_scheduler.lower()=='exponentiallr':
        #     gamma=optim_cfg.gamma if hasattr(optim_cfg, 'gamma') and optim_cfg.gamma is not None else 0.8
        #     lr_scheduler=scheduler.ExponentialLR(optimizer,gamma)
        # elif optim_cfg.lr_scheduler.lower() == 'steplr':
        #     gamma = optim_cfg.gamma if hasattr(optim_cfg, 'gamma') and optim_cfg.gamma is not None else 0.8
        #     step_size = optim_cfg.step_size if hasattr(optim_cfg, 'step_size') and optim_cfg.step_size is not None else 20
        #     lr_scheduler = scheduler.StepLR(optimizer, step_size,gamma)
        # elif optim_cfg.lr_scheduler.lower() == 'cosineannealinglr':
        #     T_max = optim_cfg.T_max if hasattr(optim_cfg, 'T_max') and optim_cfg.T_max is not None else 150
        #     eta_min = optim_cfg.eta_min if hasattr(optim_cfg, 'eta_min') and optim_cfg.eta_min is not None else 0
        #     lr_scheduler = scheduler.CosineAnnealingLR(optimizer, T_max,eta_min)
        # elif optim_cfg.lr_scheduler.lower() == 'gradualwarmupscheduler':
        #     multiplier = optim_cfg.multiplier if hasattr(optim_cfg, 'multiplier') and optim_cfg.multiplier is not None else 2
        #     warmup_epoch = optim_cfg.warmup_epoch if hasattr(optim_cfg, 'warmup_epoch') and optim_cfg.warmup_epoch is not None else 10
        #     # after_scheduler=build_scheduler(optimizer,optim_cfg.after_scheduler) if hasattr(optim_cfg, 'after_scheduler') else None
        #     after_scheduler=build_scheduler_no_optimizer(optim_cfg.after_scheduler) if hasattr(optim_cfg, 'after_scheduler') else None
        #     lr_scheduler =GradualWarmupScheduler(optimizer,multiplier=multiplier,warmup_epoch=warmup_epoch,after_scheduler=after_scheduler)
    return lr_scheduler

def build_scheduler_no_optimizer(optim_cfg):
    if optim_cfg.lr_scheduler is not None:
        if optim_cfg.lr_scheduler.lower()=='multisteplr':
            lr_scheduler=partial(scheduler.MultiStepLR,
                                 milestones=optim_cfg.milestones,
                                 gamma=optim_cfg.gamma,
                                 last_epoch=optim_cfg.last_epoch)
        elif optim_cfg.lr_scheduler.lower()=='reducelronplateau':
            mode=optim_cfg.mode if hasattr(optim_cfg, 'mode') and optim_cfg.mode is not None else 'min'
            factor=optim_cfg.factor if hasattr(optim_cfg, 'factor') and optim_cfg.factor is not None else 0.1
            patience=optim_cfg.patience if hasattr(optim_cfg, 'patience') and optim_cfg.patience is not None else 10
            verbose=optim_cfg.verbose if hasattr(optim_cfg, 'verbose') and optim_cfg.verbose is not None else False
            threshold=optim_cfg.threshold if hasattr(optim_cfg, 'threshold') and optim_cfg.threshold is not None else 0.0001
            threshold_mode=optim_cfg.threshold_mode if hasattr(optim_cfg, 'threshold_mode') and optim_cfg.threshold_mode is not None else 'rel'
            cooldown=optim_cfg.cooldown if hasattr(optim_cfg, 'cooldown') and optim_cfg.cooldown is not None else 0
            min_lr=optim_cfg.min_lr if hasattr(optim_cfg, 'min_lr') and optim_cfg.min_lr is not None else 0
            eps=optim_cfg.eps if hasattr(optim_cfg, 'eps') and optim_cfg.eps is not None else 1e-08
            lr_scheduler=partial(scheduler.ReduceLROnPlateau,
                                 mode=mode,factor=factor,patience=patience,
                                 verbose=verbose,threshold=threshold,threshold_mode=threshold_mode,
                                 cooldown=cooldown,min_lr=min_lr,eps=eps)
        elif optim_cfg.lr_scheduler.lower()=='exponentiallr':
            gamma=optim_cfg.gamma if hasattr(optim_cfg, 'gamma') and optim_cfg.gamma is not None else 0.8
            lr_scheduler=partial(scheduler.ExponentialLR,gamma=gamma)
        elif optim_cfg.lr_scheduler.lower() == 'steplr':
            gamma = optim_cfg.gamma if hasattr(optim_cfg, 'gamma') and optim_cfg.gamma is not None else 0.8
            step_size = optim_cfg.step_size if hasattr(optim_cfg, 'step_size') and optim_cfg.step_size is not None else 20
            lr_scheduler = partial(scheduler.StepLR,
                                   step_size=step_size,gamma=gamma)
        elif optim_cfg.lr_scheduler.lower() == 'cosineannealinglr':
            T_max = optim_cfg.T_max if hasattr(optim_cfg, 'T_max') and optim_cfg.T_max is not None else 150
            eta_min = optim_cfg.eta_min if hasattr(optim_cfg, 'eta_min') and optim_cfg.eta_min is not None else 0
            lr_scheduler = partial(scheduler.CosineAnnealingLR, T_max=T_max,eta_min=eta_min)
    else:
        lr_scheduler = partial(scheduler.MultiStepLR,
                               milestones=[40,80,120],gamma=0.5,last_epoch=-1)
    return lr_scheduler

if __name__=='__main__':
    model=hrnet_mv32(in_channels=3, out_channels=1)
    # opt=torch.optim.Adam(model.parameters(),0.0001)
    # lr_s=GradualWarmupScheduler(opt,10,5)
    # ## StepLR 可视化学习率
    # plot_lr(lr_s, title='StepLR')
    _cfg=RUNNING_TRAIN_CFG()
    _cfg,yaml_path=BuildingConfigFromYaml(_cfg,"cfgs_Penta_IMG_DSM_Hrnet32")
    opt=build_optimizer(model,_cfg.OPTIMIZATION_CFG,True)
    build_scheduler(opt,_cfg.OPTIMIZATION_CFG.lr_scheduler)
    print(opt.defaults)
