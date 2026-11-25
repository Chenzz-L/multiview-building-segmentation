import inspect
from functools import partial
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.datasets.data_utils import *
from data.datasets.dataset_RGB import *
from data.datasets.dataset_DSM import *
from data.datasets.dataset_RGB_MV import *
from data.datasets.dataset_DSM_MV import *
from data.datasets.dataset_DSM_Fusion_image_xyz_branch import *
from data.datasets.dataset_RGB_Fusion_image_xyz_branch import *
from data.datasets.dataset_DSM_Fusion_image_xyz_branch_two import *
from data.datasets.dataset_Fusion_image_xyz_branch_padding import *
from data.datasets.dataset_Melbourne_DSM import *
from data.datasets.dataset_VIT_viewfusion import *
from data.datasets.dataset_Hangzhou_RGB_DSM_MIX import *
from data.datasets.dataset_RGB_selectviews import *
from data.datasets.dataset_DSM_selectviews import *
from data.datasets.dataset_RGB_DSM_split_selectviews import *
from data.datasets.dataset_RGB_DSM_split import *
from data.datasets.dataset_DSM_Fusion_image_xyz_branch_selectviews import *
from data.datasets.dataset_RGB_DSM_COCO_Simple import *
from data.datasets.dataset_RGB_DSM_COCO_Simple_solov2 import *
from data.datasets.dataset_RGB_DSM_split_partdata import *
from data.datasets.dataset_RGB_DSM_COCO_Simple_solov2_partdata import *
from data.datasets.dataset_RGB_DSM_COCO_Simple_partdata import *
from data.datasets.dataset_RGB_DSM_COCO_Simple_partdata_centers import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_split_selectviews_feature import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_split_feature import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_cam_feature import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_cluster_feature import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_unified_cluster import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_RandomSelect_CrossAttn import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_sail_memory import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_raw_views import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_offline_ablation import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_sail_remote import *
from data.datasets.dataset_DSM_Fusion_RGB_DSM_offline_mean import *

from data.inference_datasets.inference_dataset_RGB import *
from data.inference_datasets.inference_dataset_DSM import *
from data.inference_datasets.inference_dataset_RGB_Fusion import *
from data.inference_datasets.inference_dataset_DSM_Fusion import *
from data.inference_datasets.inference_dataset_DSM_Fusion_image_xyz_branch import *
from data.inference_datasets.inference_dataset_DSM_Fusion_image_xyz_branch_v2 import *
from data.inference_datasets.inference_dataset_RGB_Fusion_image_xyz_branch import *
from data.inference_datasets.inference_dataset_DSM_Fusion_image_xyz_branch_oom import *
from data.inference_datasets.inference_dataset_Hangzhou_DSM import *
from data.datasets.dataste_support import *
from data.inference_datasets.inference_dataset_Hangzhou_RGB import *
from data.inference_datasets.inference_dataset_Hangzhou_RGB_DSM_split import *
from data.inference_datasets.inference_dataset_RGB_DSM_COCO_Simple import *
from data.inference_datasets.inference_dataset_DSM_Fusion_RGB_DSM_split_feature import *
from data.inference_datasets.inference_dataset_DSM_Fusion_RGB_DSM_cam_feature import *

__all__ = ['build_Dataset', 'build_DataLoader']

# def build_DataLoader(dataset, cfg,is_distributed=False):
#     #shuffle与sampler是冲突的，并行训练需要设置sampler，此时务必将shuffle设为Flase，
#     # 但是并不意味着数据不会乱序，乱序的方式交给了sampler控制
#     data_loader, data_sampler=None,None
#     if dataset is not None:
#         # if is_distributed:
#         #     if cfg.data_type.lower()=='train':
#         #         data_sampler=tdata.distributed.DistributedSampler(dataset,shuffle=cfg.shuffle,drop_last=True)
#         #     elif cfg.data_type.lower()=='test':
#         #         data_sampler=SequentialDistributedSampler(dataset,drop_last=False)
#         drop_last=getattr(cfg,'drop_last',False)
#         num_workers=getattr(cfg,'num_workers',0)
#         if "Inference" in dataset.__class__.__name__:
#             data_loader = tdata.DataLoader(dataset=dataset, batch_size=1,
#                                            num_workers=num_workers,
#                                            shuffle=False,
#                                            sampler=data_sampler, collate_fn=lambda x:x)
#         elif 'Fusion' in dataset.__class__.__name__ :
#             # data_loader = tdata.DataLoader(dataset=dataset, batch_size=cfg.batch_size,
#             #                             num_workers=num_workers,
#             #                             shuffle=cfg.shuffle if data_sampler==None else False,
#             #                             pin_memory=True,sampler=data_sampler,collate_fn=dataset_collate,
#             #                             drop_last=drop_last)
#             data_loader = tdata.DataLoader(dataset=dataset, batch_size=cfg.batch_size,
#                             num_workers=num_workers,
#                             shuffle=cfg.shuffle if data_sampler==None else False,
#                             pin_memory=True,sampler=data_sampler,collate_fn=globals()[cfg.collate_fn],
#                             drop_last=drop_last)
#         else:
#             # data_loader = tdata.DataLoader(dataset=dataset, batch_size=cfg.batch_size,
#             #                                num_workers=cfg.num_workers,
#             #                                shuffle=cfg.shuffle if data_sampler==None else False,
#             #                                pin_memory=False, sampler=data_sampler,
#             #                                drop_last= False if cfg.data_type.lower()=='test' else True)
#             drop_last= False if cfg.data_type.lower()=='test' else drop_last
#             # data_loader = tdata.DataLoader(dataset=dataset, batch_size=cfg.batch_size,
#             #                                num_workers=cfg.num_workers,
#             #                                shuffle=cfg.shuffle if data_sampler==None else False,
#             #                                pin_memory=False, sampler=data_sampler,
#             #                                drop_last= drop_last,collate_fn=globals()[getattr(cfg,'collate_fn',datalder.default_collate)])
#             data_loader = tdata.DataLoader(dataset=dataset, batch_size=cfg.batch_size,
#                                 num_workers=num_workers,
#                                 shuffle=cfg.shuffle if data_sampler==None else False,
#                                 pin_memory=False, sampler=data_sampler,drop_last= drop_last)

#     return data_loader,data_sampler

def build_DataLoader(dataset, cfg, is_distributed=False):
    data_loader, data_sampler=None,None
    if dataset is not None:
        if "Inference" in dataset.__class__.__name__:
            cfg.collate_fn = globals()[cfg.collate_fn]
            # data_loader = tdata.DataLoader(dataset = dataset, collate_fn = lambda x:x,**cfg.__dict__)
            data_loader = tdata.DataLoader(dataset = dataset, **cfg.__dict__)
        elif 'Fusion' in dataset.__class__.__name__ :
            cfg.collate_fn = globals()[cfg.collate_fn]
            # data_loader = tdata.DataLoader(dataset = dataset,collate_fn = globals()[cfg.collate_fn],**cfg.__dict__)
            data_loader = tdata.DataLoader(dataset = dataset,**cfg.__dict__)
        else:
            # cfg.collate_fn = globals()[cfg.collate_fn]
            data_loader = tdata.DataLoader(dataset = dataset, **cfg.__dict__)

    return data_loader,data_sampler

def build_Dataset(cfg):
    """
    1、根据cfg的dataset_type创建dataset
    2、inspect模块获取dataset的参数名
    3、通过参数名获取对应的参数值
    4、partial调用对象
    Args:
        cfg:

    Returns:

    """
    # print(supported_datasets.keys())
    if cfg is None:
        return None
    assert cfg.type in supported_datasets, 'unsupported dataset type: %s...' % cfg.type

    sig = inspect.signature(dict(inspect.getmembers(supported_datasets[cfg.type]))['__dict__']['__init__'])
    _={}
    for key,vlaue in sig.parameters.items():
        if key=='self' or not hasattr(cfg,key):
            continue
        _[key]=getattr(cfg,key)
    dset=partial(supported_datasets[cfg.type],**_)()
    if getattr(cfg,'set_dir',None) is not None:
        set_func=getattr(dset,"set_dir")

        partial(set_func,*[getattr(cfg,key) for key in inspect.signature(set_func).parameters.keys()])()

    if getattr(cfg, 'set_stat', None) is not None:
        set_func = getattr(dset, "set_stat")

        partial(set_func, *[getattr(cfg, key) for key in inspect.signature(set_func).parameters.keys()])()

    return dset



# if __name__=='__main__':
#     parser = argparse.ArgumentParser(description='Prediction')
#     # path
#     parser.add_argument('-type', type=str, default='InferenceDataset_RGB', help='')
#     parser.add_argument('-data_dir', type=str, default=r'/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/test/samples_colinear', help='')
#     parser.add_argument('-standardization', type=bool, default=True)
#     parser.add_argument('-img_type', type=str, default='tif')
#     parser.add_argument('-set_stat', type=bool, default=False)
#     parser.add_argument('-set_dir', type=bool, default=True)
#     parser.add_argument('-image_dir_name', type=str, default='image')
#     parser.add_argument('-label_dir_name', type=str, default='label')

#     args = parser.parse_args()
#     dataset=build_Dataset(args)
#     print("ok")
    # print(build_Dataset.__code__.co_varnames)
    # for key,value in dict(inspect.getmembers(globals()['Inference_BasicDataset'])).items():
    #     print(f'{key}:{value}')
    # sig = inspect.signature(dict(inspect.getmembers(globals()['Inference_BasicDataset']))['__dict__']['__init__'])
    # for key,value in sig.parameters.items():
    #     print(f'{key}:{value}')
    # sig = inspect.signature(build_DataLoader_v2)
    # for key,value in sig.parameters.items():
    #     print(f'{key}:{value}')
