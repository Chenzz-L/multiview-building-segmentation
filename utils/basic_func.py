# -*- coding: utf-8 -*-
import glob
import os
import torch
import argparse
import inspect

__all__ = ['create_root_dir', 'save_settings', 'resume_training', 'resume_from_checkpoint',
           'load_pretrain','resume_model_with_checkpoint','resume_optimizer','resume_lr_scheduler',
           'load_pretrain_with_checkpoint','save_model','resume_other_param_with_checkpoint','get_newest_file',
           'get_param_from_config_func']

def get_newest_file(input_dir,end='pth'):
    files=glob.glob(os.path.join(input_dir,"*."+end))

    # lists = os.listdir(tekst_report) #列出目录的下所有文件和文件夹保存到lists
    # print(list)
    # files.sort(key=lambda fn:os.path.getmtime(input_dir + "\\" + fn))  # 按时间排序
    if len(files)!=0:
        files.sort(key=lambda fn: os.path.getmtime(fn))  # 按时间排序
        file_new = os.path.join(input_dir,files[-1]) # 获取最新的文件保存到file_new
        return file_new
    return None

def get_best_metrics_file(input_dir,metric='best_val_iou',mode='max',end='pth'):
    func={'max':max,'min':min}
    files=glob.glob(os.path.join(input_dir,"*."+end))
    best_metric=-float('inf') if mode=='max' else float('inf')
    best_file=None
    for file in files:
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        assert metric in checkpoint.keys(),'key not in the file'
        metric_data=checkpoint[metric]
        best_metric=func[mode](best_metric,metric_data)
        if best_metric==metric_data:
            best_file=file

    return best_file

def create_root_dir(out_file):
    assert out_file is not None
    assert out_file != ''
    out_dir, _ = os.path.split(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def save_settings(args: argparse.ArgumentParser, output_dir: str, open_mode='a+'):
    r"""
    :param output_dir:
    :param args:
    :param open_mode:
    """
    assert output_dir is not None
    assert output_dir != ''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save settings
    log_txt = open(output_dir + '/settings.txt', open_mode)
    for arg in vars(args):
        print(arg, getattr(args, arg))
        log_txt.write(arg + ' : ' + str(getattr(args, arg)) + '\n')
    log_txt.close()

def save_model(model, optimizer, lr_scheduler, out_file, **kwargs):
    assert model is not None
    model_dict = {'model': model.state_dict()}
    if optimizer is not None:
        model_dict['optimizer'] = optimizer.__class__.__name__
        model_dict['optimizer_state'] = optimizer.state_dict()
    if lr_scheduler is not None:
        model_dict['lr_scheduler'] = lr_scheduler.__class__.__name__
        model_dict['lr_scheduler_state'] = lr_scheduler.state_dict()
    for key, values in kwargs.items():
        model_dict[key] = values
    torch.save(model_dict, out_file)

def resume_from_checkpoint(model, cfg):
    print('Resuming training from checkpoint ...... ')
    assert os.path.isfile(cfg.checkpoint_file)
    checkpoint = torch.load(cfg.checkpoint_file)
    model.load_state_dict(checkpoint['model'])
    cfg.start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint.keys() else 0
    cfg.best_loss = checkpoint['best_loss'] if 'best_loss' in checkpoint.keys() else float('inf')
    cfg.best_val_iou = checkpoint['best_val_iou'] if 'best_val_iou' in checkpoint.keys() else float('-inf')

def resume_training(model, optimizer, lr_scheduler, args):
    print('Resuming training from checkpoint ...... ')
    checkpoint = torch.load(args.trained_checkpoint)
    model.load_state_dict(checkpoint['model'])
    if args.resume_optimizer and optimizer is not None:
        print('resuming optimizer:', optimizer.__class__.__name__)
        assert optimizer.__class__.__name__ == checkpoint['optimizer']
        # if checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if args.resume_lr_scheduler and lr_scheduler is not None:
        print('resuming lr_scheduler:', lr_scheduler.__class__.__name__)
        assert lr_scheduler.__class__.__name__ == checkpoint['lr_scheduler']
        # if checkpoint['lr_scheduler'] is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
    args.start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint.keys() else 0
    args.best_loss = checkpoint['best_loss'] if 'best_loss' in checkpoint.keys() else float('inf')
    args.best_val_iou = checkpoint['best_val_iou'] if 'best_val_iou' in checkpoint.keys() else float('-inf')
    args.global_counter = checkpoint['global_counter'] if 'global_counter' in checkpoint.keys() else 0

def load_pretrain(model,args):
    print('loading model state for pretrain')
    load_pretrain_with_checkpoint(model,args.checkpoint_file,args.weight_name)
    # checkpoint = torch.load(args.checkpoint_file, map_location=torch.device('cpu'))
    # # 去除可能不在模型中的参数
    # # weights = checkpoint['model']
    # weights = checkpoint[args.weight_name] if args.weight_name is not None else checkpoint['model']
    # model_dict = model.state_dict()
    # weights={k:v for k,v in weights.items() if k in model_dict}

    # model_dict.update(weights)

    # model.load_state_dict(model_dict)

def load_pretrain_with_checkpoint(model,checkpoint_file,weight_name='model'):
    if os.path.isdir(checkpoint_file):
        checkpoint_file=get_newest_file(checkpoint_file)
    
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

    # None表示只有权重
    if weight_name:
        weights = checkpoint[weight_name]
    else:
        weights=checkpoint
    loaded=False
    # 去除可能不在模型中的参数
    model_dict = model.state_dict()
    pretrain_weights={}
    for k,v in weights.items():
        if k in model_dict:
            #只考虑模型权重比预训练权重尺寸大的情况
            if v.shape!=model_dict[k].shape:
                temp_weight=model_dict[k].clone()
                batch_slice=slice(0,v.shape[0])
                channel_slice=slice(0,v.shape[1])
                h_slice=slice(0,v.shape[2])
                w_slice=slice(0,v.shape[3])
                temp_weight[batch_slice,channel_slice,h_slice,w_slice]=v.data
                pretrain_weights[k]=temp_weight
            else:
                pretrain_weights[k]=v
            loaded=True
    # weights = {k: v for k, v in weights.items() if k in model_dict}
    # model_dict.update(weights)
    model_dict.update(pretrain_weights)
    model.load_state_dict(model_dict)
    if loaded:
        print('load weights successful.....')

def resume_model_with_checkpoint(model,args):
    print('Resuming training from checkpoint ...... ')
    load_pretrain(model,args)
    # checkpoint = torch.load(args.checkpoint_file,map_location=torch.device('cpu'))
    # # checkpoint = torch.load(args.checkpoint_file,map_location='cuda:{}'.format(0))
    # #去除可能不在模型中的参数
    # weights=checkpoint['model']
    # model_dict=model.state_dict()
    # weights={k:v for k,v in weights.items() if k in model_dict}
    # model_dict.update(weights)
    # model.load_state_dict(model_dict)

def resume_other_param_with_checkpoint(args):
    """
    加载checkpoints中的其他参数，DPP中需要每一个程序都调用
    Args:
        args:

    Returns:

    """
    checkpoint = torch.load(args.checkpoint_file, map_location=torch.device('cpu'))
    args.start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else 0
    args.epochs += args.start_epoch
    args.best_loss = checkpoint['best_loss'] if 'best_loss' in checkpoint.keys() else float('inf')
    args.best_val_iou = float(checkpoint['best_val_iou']) if 'best_val_iou' in checkpoint.keys() else float('-inf')
    args.global_counter = checkpoint['global_counter'] if 'global_counter' in checkpoint.keys() else 0

def resume_optimizer(optimizer,args):
    checkpoint = torch.load(args.checkpoint_file,map_location=torch.device('cpu'))
    if args.resume_optimizer and optimizer is not None:
        print('resuming optimizer:', optimizer.__class__.__name__)
        assert optimizer.__class__.__name__ == checkpoint['optimizer']
        if isinstance(args.lr, (list, tuple)):
            assert len(checkpoint['optimizer_state']['param_groups'])==len(args.lr)
            for i in range(len(checkpoint['optimizer_state']['param_groups'])):
                checkpoint['optimizer_state']['param_groups'][i]['initial_lr'] = args.lr[i]
                checkpoint['optimizer_state']['param_groups'][i]['lr'] = args.lr[i]
        else:
            checkpoint['optimizer_state']['param_groups'][0]['initial_lr']=args.lr
            checkpoint['optimizer_state']['param_groups'][0]['lr']=args.lr

        optimizer.load_state_dict(checkpoint['optimizer_state'])

def resume_lr_scheduler(lr_scheduler,args):
    checkpoint = torch.load(args.checkpoint_file,map_location=torch.device('cpu'))
    if args.resume_lr_scheduler and lr_scheduler is not None:
        print('resuming lr_scheduler:', lr_scheduler.__class__.__name__)
        assert lr_scheduler.__class__.__name__ == checkpoint['lr_scheduler']
        # if checkpoint['lr_scheduler'] is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

def get_param_from_config_func(func, cfg):
    assert inspect.isfunction(func)
    sing_func = inspect.signature(func)
    param_dict = {}
    for name, param in sing_func.parameters.items():
        # 这个对dict不起作用
        # if hasattr(cfg, name):
        #     param_dict[name] = getattr(cfg, name)
        if name in cfg:
            param_dict[name] = cfg[name]
    return param_dict

# if __name__=='__main__':
    # checkpoint_file='/path/to/user/NAS/gwx/experiment_result/20221207/2022-12-07-10-17-08-storage-upernet_swin_tiny_patch4_window7_512x512.pth'
    # checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    # weight=checkpoint.pop('state_dict')
    # checkpoint['model']=weight
    # torch.save(checkpoint,'/path/to/user/NAS/gwx/experiment_result/all_swin_experiment/pretrained.pth')
