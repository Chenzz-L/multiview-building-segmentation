from abc import ABC,abstractmethod
from typing import Tuple,List

import yaml
# from ruamel import yaml
from yacs.config import CfgNode as CN
from easydict import EasyDict as edict

__all__=['RUNNING_TRAIN_CFG','RUNNING_PREDICT_CFG','write_yaml','load_yaml','convert_CfgNode_2_dict']
def load_yaml(file):
    with open(file, mode='r', encoding='utf-8') as fd:
        data = yaml.load(fd, Loader=yaml.FullLoader)
        # data = yaml.load(fd, Loader=yaml.Loader)
    return data

def write_yaml(path,CfgNode):
    yaml_data = convert_CfgNode_2_dict(CfgNode)
    with open(path, encoding='utf-8',mode='w') as f:
        try:
            # yaml.safe_dump(data=yaml_data,stream=f,allow_unicode=True)
            yaml.dump(data=yaml_data,stream=f,allow_unicode=True)
        except Exception as e:
            print(e)

def convert_CfgNode_2_dict(cfgnode:CN or edict):
    data=dict(cfgnode)
    for k,v in data.items():
        if isinstance(v,(CN,edict)):
            data[k]=convert_CfgNode_2_dict(v)
        if isinstance(v,(Tuple,List)) and isinstance(v[0],(CN,edict)):
            t_v=[]
            for vv in v:
                t_v.append(convert_CfgNode_2_dict(vv))
            data[k]=t_v
    return data

class AbstractNode(ABC):
    def __init__(self):
        super(AbstractNode, self).__init__()
        self.C=CN()

    def get_cfg(self):
        return self.C.clone()

    def merge_from_file(self,path):
        self.C.merge_from_file(path)

class DATASET_CFG(AbstractNode):
    def __init__(self):
        super(DATASET_CFG, self).__init__()
        self.C.dataset_type='BasicDataset'
        self.C.data_dir=''
        self.C.augmentation=True
        # self.C.get_edge_from_file=False
        self.C.standardization=True
        self.C.set_dir=True
        self.C.set_stat=False
        self.C.balance_samples=False
        self.C.image_dir_name='image'
        self.C.label_dir_name='label'
        self.C.clip=None #if you clip ,input must is tuple[int,int] or list[int,int]
        self.C.data_pipeline=None

class DATALOADER_CFG(AbstractNode):
    def __init__(self):
        super(DATALOADER_CFG, self).__init__()
        self.C.batch_size = 16
        self.C.num_workers = 4
        self.C.shuffle = True
        self.C.data_type = 'train'

class MODEL_CFG(AbstractNode):
    def __init__(self):
        super(MODEL_CFG, self).__init__()
        # self.C.model_name='hrnet32'
        # self.C.from_str = True
        # self.C.in_channels = 3
        # self.C.out_channels = 1
        # self.C.from_config = False
        self.C.num_class = 2
        self.C.dropout_ratio = 0.1

class OPTIMIZATION_CFG(AbstractNode):
    def __init__(self):
        super(OPTIMIZATION_CFG, self).__init__()
        self.C.optimizer = 'Adam'
        self.C.parmas_layers=None #only for lr is tuple or list,multiple optimizers
        self.C.lr=1e-4
        self.C.lr_scheduler = 'reducelronplateau'

class TRAINER_CFG(AbstractNode):
    def __init__(self):
        super(TRAINER_CFG, self).__init__()
        self.C.model_name = 'hrnet'
        self.C.exp_name = 'experiment_dsm_affine'
        self.C.dataset_type = 'BasicDataset'
        self.C.out_dir = "/path/to/user/NAS/gwx/experiment_result/20221013/"
        self.C.tensorboard_dir='tensorboard'
        self.C.debug=False
        self.C.amp=True
        self.C.gradient_accumulate = True
        self.C.grad_accumulate_step = 8
        self.C.log_flag = 'iteration'
        self.C.log_interval = 10
        self.C.eval_and_save_flag = 'epoch'
        self.C.eval_and_save_interval = 1
        self.C.start_epoch = 0
        self.C.epochs = 300
        self.C.batch_size = 16
        self.C.global_stats_count = 1
        self.C.checkpoint_file=None
        self.C.weight_name= 'model'
        self.C.resume_from_checkpoint = False
        self.C.resume_optimizer=False
        self.C.resume_lr_scheduler=False
        self.C.pretrain=False
        self.C.freeze_bn=False
        self.C.freeze_model=False
        self.C.lr=[1e-6]
        self.C.TRAIN_TYPE = ['binary', 'mv_hrnet'][0]

class PREDICT_NODE(AbstractNode):
    def __init__(self):
        super(PREDICT_NODE, self).__init__()
        self.C.checkpoint_file=None
        self.C.out_dir=None
        # prediction and evaluation
        self.C.boundary_buffers=(0, 1, 2, 3, 4)
        self.C.set_out_branch=1
        # predicting setting
        self.C.visual_feature=False
        self.C.visual_layers=("last_layer", "xyz_branch")
        self.C.scale_list=[1]
        self.C.crop_width=512
        self.C.crop_height=512
        self.C.crop_stride=170
        self.C.threshold=0.50
        self.C.debug=True
        self.C.run_type='binary'#'binary', 'hrnet_mv'

class RUNNNING_CFG(ABC):
    def __init__(self):
        self.cfg=CN()
    def get_cfg(self):
        return self.cfg.clone()

    def clone(self):
        return self.cfg.clone()

    def clear(self):
        self.cfg.clear()

    def merge_from_file(self,path):
        yaml_data=load_yaml(path)
        #遍历yaml_data字典的键，若键存在这更新CN中的对应的键，不存在则创建
        # for yaml_key,yaml_value in yaml_data.items():
        #     if yaml_key in self.cfg.keys():
        #         for key,value in yaml_value.items():
        #             if isinstance(value, dict):
        #                 self.cfg[yaml_key][key]=edict(value)
        #             else:
        #                 self.cfg[yaml_key][key]=value

        for yaml_key,yaml_value in yaml_data.items():
            self.update_cfgnode(yaml_key,yaml_value,self.cfg)

    def merge_from_list(self,cfg_list):
        self.cfg.merge_from_list(cfg_list)

    def merge_from_other_cfg(self,othCfg):
        self.cfg.merge_from_otherCfg(othCfg)

    def freeze(self):
        self.cfg.freeze()

    def update_cfgnode(self, key, value, cfgNode):

        if key in cfgNode.keys():
            if value==None:
                cfgNode[key]=None
            else:
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        self.update_cfgnode(sub_key, sub_value, cfgNode[key])
                    else:
                        cfgNode[key][sub_key] = sub_value
        else:
            cfgNode[key]=edict(value)

    def dump(self):
        self.cfg.dump()

class RUNNING_TRAIN_CFG(RUNNNING_CFG):
    def __init__(self):
        super(RUNNING_TRAIN_CFG, self).__init__()
        self.cfg.OPTIMIZATION_CFG=OPTIMIZATION_CFG().get_cfg()
        self.cfg.TRAIN_DATASET_CFG=DATASET_CFG().get_cfg()
        self.cfg.TRAIN_DATALOADER_CFG=DATALOADER_CFG().get_cfg()
        self.cfg.VAL_DATASET_CFG=DATASET_CFG().get_cfg()
        self.cfg.VAL_DATALOADER_CFG=DATALOADER_CFG().get_cfg()
        self.cfg.TRAINER_CFG=TRAINER_CFG().get_cfg()
    # def __init__(self):
    #     self.cfg=CN()
    #     self.cfg.MODEL_CFG=MODEL_CFG().get_cfg()
    #     self.cfg.OPTIMIZATION_CFG=OPTIMIZATION_CFG().get_cfg()
    #     self.cfg.TRAIN_DATASET_CFG=DATASET_CFG().get_cfg()
    #     self.cfg.TRAIN_DATALOADER_CFG=DATALOADER_CFG().get_cfg()
    #     self.cfg.VAL_DATASET_CFG=DATASET_CFG().get_cfg()
    #     self.cfg.VAL_DATALOADER_CFG=DATALOADER_CFG().get_cfg()
    #     self.cfg.TRAINER_CFG=TRAINER_CFG().get_cfg()

    # def get_cfg(self):
    #     return self.cfg.clone()

    # def clone(self):
    #     return self.cfg.clone()

    # def clear(self):
    #     self.cfg.clear()

    # def merge_from_file(self,path):
    #     yaml_data=load_yaml(path)
    #     #遍历yaml_data字典的键，若键存在这更新CN中的对应的键，不存在则创建
    #     # for yaml_key,yaml_value in yaml_data.items():
    #     #     if yaml_key in self.cfg.keys():
    #     #         for key,value in yaml_value.items():
    #     #             if isinstance(value, dict):
    #     #                 self.cfg[yaml_key][key]=edict(value)
    #     #             else:
    #     #                 self.cfg[yaml_key][key]=value

    #     for yaml_key,yaml_value in yaml_data.items():
    #         self.update_cfgnode(yaml_key,yaml_value,self.cfg)

    # def merge_from_list(self,cfg_list):
    #     self.cfg.merge_from_list(cfg_list)

    # def merge_from_other_cfg(self,othCfg):
    #     self.cfg.merge_from_otherCfg(othCfg)

    # def freeze(self):
    #     self.cfg.freeze()

    # def update_cfgnode(self, key, value, cfgNode):

    #     if key in cfgNode.keys():
    #         if value==None:
    #             cfgNode[key]=None
    #         else:
    #             for sub_key, sub_value in value.items():
    #                 if isinstance(sub_value, dict):
    #                     self.update_cfgnode(sub_key, sub_value, cfgNode[key])
    #                 else:
    #                     cfgNode[key][sub_key] = sub_value
    #     else:
    #         cfgNode[key]=edict(value)

    # def dump(self):
    #     self.cfg.dump()


class RUNNING_PREDICT_CFG(RUNNNING_CFG):
    def __init__(self):
        super(RUNNING_PREDICT_CFG, self).__init__()
        self.cfg.PREDICT_CFG=PREDICT_NODE().get_cfg()
        self.cfg.DATASET_CFG=DATASET_CFG().get_cfg()
        self.cfg.DATALOADER_CFG=DATALOADER_CFG().get_cfg()



if __name__ == "__main__":
    cc=RUNNING_PREDICT_CFG()
    cc.merge_from_file('/path/to/user/NAS/gwx/code/cnncode/cfgs/predcit_dom_SwinT_Upernet.yaml')
    write_yaml('/path/to/user/NAS/gwx/code/cnncode/cfgs/11.yaml',cc.get_cfg())
