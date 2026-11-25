import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Iterable
import torch
import torch.nn as nn
from models.hrnet_mv import *
# __all__ = ['freeze_bn','freeze_bn_model','freeze_by_names','unfreeze_by_names']
__all__ = ['freeze_by_names','unfreeze_by_names']

# def freeze_bn(ly):
#     classname = ly.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         for param in ly.parameters():
#             param.requires_grad = False
#         ly.eval()

# def freeze_bn_model(model:torch.nn.Module,except_layer=()):
#     for name,module in model.named_children():
#         if name  in except_layer:
#             classname = module.__class__.__name__
#             if classname.find('BatchNorm') != -1:
#                 for param in module.parameters():
#                     param.requires_grad = False
#                 module.eval()
#             elif isinstance(module,torch.nn.Module):freeze_bn_model(module)
# def freeze_bn(ly,):
#     if isinstance(ly, torch.nn.modules.batchnorm._BatchNorm):
#         for param in ly.parameters():
#             param.requires_grad = False
#         ly.eval()
#         # ly.affine=False
#         ly.track_running_stats=False
#     elif isinstance(ly,(nn.Sequential,nn.ModuleList,nn.ModuleDict)) or len(ly._modules)>0:
#         for name,module in ly.named_children():
#             freeze_bn(module)
            
# def freeze_bn_model(model:torch.nn.Module,except_layer=()):
#     for name,module in model.named_children():
#         if name  in except_layer:
#             freeze_bn(module)
           

def freeze_bn(ly,freeze=True):
    if isinstance(ly, torch.nn.modules.batchnorm._BatchNorm):
        for param in ly.parameters():
            param.requires_grad = not freeze
        # if freeze:
        #     ly.eval()
        # ly.affine=not freeze
        # ly.training=not freeze
        ly.track_running_stats= not freeze
    elif isinstance(ly,(nn.Sequential,nn.ModuleList,nn.ModuleDict)) or len(ly._modules)>0:
        for name,module in ly.named_children():
            freeze_bn(module,freeze)
           
def set_freeze_bn_by_name(model:torch.nn.Module,except_layer=(), freeze=True):
    for name, child in model.named_children():
        if any(name==s for s in except_layer):
            freeze_bn(child,freeze)
        elif any(name in s for s in except_layer):
            new_freeze_names=['.'.join(name.split('.')[1:]) for name in except_layer]
            set_freeze_bn_by_name(child,new_freeze_names,freeze)
        else:
            freeze_bn(child, not freeze)

def unfreeze_bn_model(model:torch.nn.Module,except_layer=()):
    set_freeze_bn_by_name(model,except_layer,freeze=False)

def freeze_bn_model(model:torch.nn.Module,except_layer=()):
    set_freeze_bn_by_name(model,except_layer,freeze=True)

def set_freeze_by_names(model, layer_names, freeze=True):
    print('freeze the model by name.....')
    freezed=False
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]

    freezed=freeze_loop(model,layer_names,freeze)
    if freezed and freeze:
        print('successfully freezed the model by name.....')
    if freezed and not freeze:
        print('successfully unfreezed the model by name.....')

    # for name, child in model.named_children(): 
    #     if name not in layer_names:
    #         continue
    #     for param in child.parameters():
    #         param.requires_grad = not freeze
    #     freezed=True
    # if freezed:
    #     print('successfully freezed the model by name.....')

def freeze_loop(model,freeze_names,freeze):
    freezed=False
    for name, child in model.named_children():
        if any(name==s for s in freeze_names):
            for param in child.parameters():
                param.requires_grad = not freeze
                freezed=True
        elif any(name in s for s in freeze_names):
            new_freeze_names=['.'.join(name.split('.')[1:]) for name in freeze_names]
            freezed=freeze_loop(child,new_freeze_names,freeze)
        else:
            for param in child.parameters():
                param.requires_grad = freeze
    return freezed

def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)
    # freeze_bn_model(model,layer_names)
    set_freeze_bn_by_name(model,layer_names,freeze=True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)
    # unfreeze_bn_model(model,layer_names)
    set_freeze_bn_by_name(model,layer_names,freeze=False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)
    
    
if __name__=='__main__':
    model=hrnet_mv32(in_channels=3, out_channels=1)
    freeze_name=['conv1','bn1','conv2','bn2','layer1',
                 'transition1','stage2','transition2',
                 'stage3','transition3','stage4','last_layer']
    freeze_by_names(model,freeze_name)
    for name, child in model.named_children():
        print(name)
        for param in child.parameters():
            print(param.requires_grad) 
    print("ok")
