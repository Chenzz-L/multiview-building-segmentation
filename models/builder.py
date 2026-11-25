import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Dict,List,Tuple
import inspect

# from torchsummary import summary

# encoder
from models.resnet import *
from models.effnet_config import *
from models.effnetv2 import *
# decoder
from models.fpn import *
from models.refinenet import *
from models.deeplabv3plus import *
from models.danet import *
# net
from models.hrnet import *
from models.hrnet_mv import *
from models.hrnet_fusion_test import *
from models.hrnet_fusion_stage2_proj import *
from models.Hrnet_image_xyz_branch import *
from models.Hrnet_image_xyz_branch_two import *
from models.brnet import *
from models.epunet import *
# from models.encoder_decoder import *
from models.mvmapper_framer import *
from models.segformer_spatialfusion import SegFormerSpatialFusion,SegFormer
from models.backbones import *
from models.heads import *
from models.necks import *
import os
import importlib.util
from models.heads.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from detectron2.layers import ShapeSpec 

# def import_all_modules_from_path(path_to_modules):
#     # 遍历指定目录下的所有文件
#     for filename in os.listdir(path_to_modules):
#         # 检查文件是否为Python文件
#         if filename.endswith(".py") and not filename.startswith("__") and filename != "builder.py":
#             # 构建模块名
#             module_name = filename[:-3]
#             # 构建模块的完整路径
#             module_path = os.path.join(path_to_modules, filename)
#             # 导入模块
#             spec = importlib.util.spec_from_file_location(module_name, module_path)
#             module = importlib.util.module_from_spec(spec)
#             spec.loader.exec_module(module)
#             # 将模块添加到全局命名空间中
#             globals()[module_name] = module

# import_all_modules_from_path(os.path.dirname(os.path.abspath(__file__)))

__all__ = [
    'generate_model_from_configs', 'generate_model_from_str', 'generate_model',
    'hrnet18', 'hrnet32', 'hrnet48','hrnet_mv18','hrnet_mv32','hrnet_mv48',
    'BRNet', 'BRNetv2', 'UNet',
    'EPUNet',
    'ResNet_FPN', 'ResNet_FPNDA', 'ResNet_RefinePSP', 'ResNet_RefineDA', 'ResNet_Deeplabv3Plus',
    'EffNetV2_FPN', 'EffNetV2_FPNDA', 'EffNetV2_RefinePSP', 'EffNetV2_RefineDA', 'EffNetV2_Deeplabv3Plus',
    'SegFormer','SegFormerSpatialFusion'

]

resnet_basic_deeplab_channels = {'18': 256, '34': 256, '50': 512, '101': 512, '152': 512, }
effnetv2_basic_fpn_channels = {'xs': 64, 's': 64, 'm': 128, 'l': 128, 'xl': 128, }
effnetv2_basic_deeplab_channels = {'xs': 256, 's': 256, 'm': 512, 'l': 512, 'xl': 512, }


# main models
def ResNet_FPN(net_size, in_channels, out_channels, basic_channels=None, **kwargs):
    from utils.basic_func import load_pretrain_with_checkpoint
    encoder = ResNet_Encoder(net_size=net_size, in_channels=in_channels)
    # load_pretrain_with_checkpoint(encoder,r'/path/to/user/NAS/gwx/code/cnncode/resnet50.pth',None)
    decoder = FPN_Decoder(in_channels=encoder.channels[::-1],
                          feature_indices=[i for i in range(len(encoder.channels))][::-1],
                          basic_channels=encoder.channels[0] if basic_channels is None else basic_channels,
                          out_channels=out_channels)
    return SegmentationEncoderDecoder(encoder, decoder, **kwargs)

def ResNet_FPNDA(net_size, in_channels, out_channels, basic_channels=None, **kwargs):
    encoder = ResNet_Encoder(net_size=net_size, in_channels=in_channels)
    decoder = FPNDA_Decoder(in_channels=encoder.channels[::-1],
                            feature_indices=[i for i in range(len(encoder.channels))][::-1],
                            basic_channels=encoder.channels[0] if basic_channels is None else basic_channels,
                            out_channels=out_channels)
    return SegmentationEncoderDecoder(encoder, decoder, **kwargs)

def ResNet_RefinePSP(net_size, in_channels, out_channels, **kwargs):
    encoder = ResNet_Encoder(net_size=net_size, in_channels=in_channels)
    decoder = RefinePSP_Decoder(in_channels=encoder.channels[::-1],
                                feature_indices=[i for i in range(len(encoder.channels))][::-1],
                                out_channels=out_channels,
                                mid_channels=None,
                                RefineConvNum=2)
    return SegmentationEncoderDecoder(encoder, decoder, **kwargs)

def ResNet_RefineDA(net_size, in_channels, out_channels, **kwargs):
    encoder = ResNet_Encoder(net_size=net_size, in_channels=in_channels)
    decoder = RefineDA_Decoder(in_channels=encoder.channels[::-1],
                               feature_indices=[i for i in range(len(encoder.channels))][::-1],
                               out_channels=out_channels,
                               mid_channels=None,
                               RefineConvNum=2)
    return SegmentationEncoderDecoder(encoder, decoder, **kwargs)

def ResNet_Deeplabv3Plus(net_size, in_channels, out_channels, **kwargs):
    encoder = ResNet_Encoder(net_size=net_size, in_channels=in_channels)
    decoder = DeepLabv3Plus_Decoder(in_channels=(encoder.channels[-1], encoder.channels[0]),
                                    feature_indices=(len(encoder.channels) - 1, 0),
                                    basic_channels=resnet_basic_deeplab_channels[net_size],
                                    out_channels=out_channels)
    return SegmentationEncoderDecoder(encoder, decoder, **kwargs)

def EffNetV2_FPN(net_size, in_channels, out_channels, basic_channels=None, **kwargs):
    encoder = EffNetV2_Encoder(net_size=net_size, in_channels=in_channels)
    decoder = FPN_Decoder(in_channels=encoder.channels[1:][::-1],
                          feature_indices=[i for i in range(1, len(encoder.channels))][::-1],
                          basic_channels=effnetv2_basic_fpn_channels[net_size] if basic_channels is None
                          else basic_channels,
                          out_channels=out_channels)
    return SegmentationEncoderDecoder(encoder, decoder, **kwargs)

def EffNetV2_FPNDA(net_size, in_channels, out_channels, basic_channels=None, **kwargs):
    encoder = EffNetV2_Encoder(net_size=net_size, in_channels=in_channels)
    decoder = FPNDA_Decoder(in_channels=encoder.channels[1:][::-1],
                            feature_indices=[i for i in range(1, len(encoder.channels))][::-1],
                            basic_channels=effnetv2_basic_fpn_channels[net_size] if basic_channels is None
                            else basic_channels,
                            out_channels=out_channels)
    return SegmentationEncoderDecoder(encoder, decoder, **kwargs)

def EffNetV2_RefinePSP(net_size, in_channels, out_channels, basic_channels=None, **kwargs):
    encoder = EffNetV2_Encoder(net_size=net_size, in_channels=in_channels)
    decoder = RefinePSP_Decoder(
        in_channels=[cfg[1] for cfg in encoder.cfgs][1:][::-1],
        feature_indices=[i for i in range(1, len(encoder.channels))][::-1],
        out_channels=out_channels,
        mid_channels=[effnetv2_basic_fpn_channels[net_size] for _ in range(len(encoder.cfgs) - 2)]
        if basic_channels is None else basic_channels,
        RefineConvNum=2)
    return SegmentationEncoderDecoder(encoder, decoder, **kwargs)

def EffNetV2_RefineDA(net_size, in_channels, out_channels, basic_channels=None, **kwargs):
    encoder = EffNetV2_Encoder(net_size=net_size, in_channels=in_channels)
    decoder = RefineDA_Decoder(
        in_channels=[cfg[1] for cfg in encoder.cfgs][1:][::-1],
        feature_indices=[i for i in range(1, len(encoder.channels))][::-1],
        out_channels=out_channels,
        mid_channels=[effnetv2_basic_fpn_channels[net_size] for _ in range(len(encoder.cfgs) - 2)]
        if basic_channels is None else basic_channels,
        RefineConvNum=2)
    return SegmentationEncoderDecoder(encoder, decoder, **kwargs)

def EffNetV2_Deeplabv3Plus(net_size, in_channels, out_channels, **kwargs):
    encoder = EffNetV2_Encoder(net_size=net_size, in_channels=in_channels)
    decoder = DeepLabv3Plus_Decoder(in_channels=(encoder.channels[-1], encoder.channels[1]),
                                    feature_indices=(len(encoder.channels) - 1, 1),
                                    basic_channels=effnetv2_basic_deeplab_channels[net_size],
                                    out_channels=out_channels)
    return SegmentationEncoderDecoder(encoder, decoder, **kwargs)

def generate_model_from_configs():
    raise NotImplementedError

def generate_model_from_str(cfg):
    """
    给定一个模型的名字, 从这个模型的名字直接生成对应的分割模型：
    模型名字包含三个部分,  encoder, net_size, decoder, 各部分之间以下划线 _ 分割
    如: ResNet_18_FPN
    :param model_name: (str)
    :return:
    """
    # 假如文件中已有该模型，直接返回该模型
    from functools import partial
    cfg_ = cfg.copy()
    if 'type' not in cfg_:
        raise KeyError('the cfg dict must contain the key "type"')

    model_name = cfg_.pop('type')
    # model = getattr(__import__('__main__'), model_name)
    if model_name in globals():
        model = globals().get(model_name)
        params = inspect.signature(model.__init__).parameters
        params = [param for param in params if param != 'self']
        if params == ['args', 'kwargs']:
            return model(**cfg_)
        cfg_ = {param: value for param, value in cfg_.items() if param in params}
        return model(**cfg_)

    else:  # 假如不存在，尝试生成该模型
        # 判断str是否符合标准
        assert len(model_name.split('_')) == 3
        encoder, net_size, decoder = model_name.split('_')
        # Encoder = encoder + '_Encoder'
        # Decoder = decoder + '_Decoder'
        # assert Encoder in globals()
        # assert Decoder in globals()
        net = encoder + '_' + decoder
        assert net in globals()
        return partial(globals()[net], net_size=net_size,**cfg_)()

# def generate_model(cfg:Dict):
#     cfg_=cfg.copy()
#     if 'type' in cfg_:
#         model = generate_model_from_str(cfg_)
#     else:
#         # pop_names=('backbone','neck','decode_head')
#         assert 'num_class' in cfg_ , '"SegEncoderDecoder" must contain "num_class" arg'

#         pop_names=('encoder', 'decoder','backbone','decode_head','spatial_fusion')

#         # if pop_names[0] not in cfg_ or pop_names[1] not in cfg_:
#         #         raise KeyError('the cfg dict must contain the key "encoder" and "decoder"')
#         model={}
#         for name in pop_names:
#             module_cfg=cfg_.pop(name,False)
#             if not module_cfg:
#                 continue
#             if isinstance(module_cfg,(Tuple,List)):
#                 module=[]
#                 for module_cfg_i in module_cfg:
#                     module.append(generate_model_from_str(module_cfg_i))
#             else:
#                 module=generate_model_from_str(module_cfg)
#             model[name]=module
#         if 'backbone' in model:
#             model['encoder']=model.pop('backbone')
#         if 'decode_head' in model:
#             model['decoder']=model.pop('decode_head')

#         if 'spatial_fusion' in model:
#             model=SegEncoderDecoderFusion(**model,**cfg_)
#         else:
#             model=SegEncoderDecoder(**model,**cfg_)
#     return model

def generate_encoder_decoder(cfg:Dict):
    cfg_ = cfg.copy()
    encoder_cfg = cfg_.pop('encoder',None)
    decoder_cfg = cfg_.pop('decoder',None)
    neck_cfg = cfg_.pop('neck', None)
    pixel_decoder_cfg = cfg_.pop('pixel_decoder', None) 
    if encoder_cfg is None or decoder_cfg is None:
        raise KeyError('the cfg dict must contain the key "encoder" and "decoder"')
    encoder = generate_model_from_str(encoder_cfg)
    neck = None
    neck_out_channels = 256
    if neck_cfg is not None:
        neck = generate_model_from_str(neck_cfg)
        if hasattr(neck, 'output_shape'):
             pass 
    
        if 'embed_dim' in neck_cfg:
            neck_out_channels = neck_cfg['embed_dim']
        elif 'out_channels' in neck_cfg:
            neck_out_channels = neck_cfg['out_channels']

    pixel_decoder = None
    if pixel_decoder_cfg is not None:
        # 构建 input_shape
        input_shape = {
            "res2": ShapeSpec(channels=neck_out_channels, stride=4),
            "res3": ShapeSpec(channels=neck_out_channels, stride=8),
            "res4": ShapeSpec(channels=neck_out_channels, stride=16),
            "res5": ShapeSpec(channels=neck_out_channels, stride=32),
        }
        
        pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            **pixel_decoder_cfg
        )
    decoder = generate_model_from_str(decoder_cfg)
    return EncoderDecoder(encoder, decoder,neck=neck,pixel_decoder=pixel_decoder, **cfg_)

def generate_encoder_decoder_crossattn(cfg:Dict):
    """构建 EncoderDecoderCrossAttn 模型（带交叉注意力融合）"""
    cfg_ = cfg.copy()
    encoder_cfg = cfg_.pop('encoder', None)
    decoder_cfg = cfg_.pop('decoder', None)
    neck_cfg = cfg_.pop('neck', None)
    cross_attn_cfg = cfg_.pop('cross_attn', None)

    if encoder_cfg is None or decoder_cfg is None:
        raise KeyError('the cfg dict must contain the key "encoder" and "decoder"')

    encoder = generate_model_from_str(encoder_cfg)
    neck = None
    if neck_cfg is not None:
        neck = generate_model_from_str(neck_cfg)

    decoder = generate_model_from_str(decoder_cfg)
    return EncoderDecoderCrossAttn(
        encoder, decoder, neck=neck,
        cross_attn_cfg=cross_attn_cfg, **cfg_
    )

def generate_normal_model(cfg:Dict):
    return generate_model_from_str(cfg)

def genrate_spatial_fusion(cfg:Dict):
    return generate_model_from_str(cfg)

def generate_mvmapper(cfg:Dict):
    cfg_ = cfg.copy()
    image_branch_cfg = cfg_.pop('image_branch',None)
    xyz_branch_cfg = cfg_.pop('xyz_branch',None)
    spatial_fusion_cfg = cfg_.pop('spatial_fusion',None)

    if image_branch_cfg is None or xyz_branch_cfg is None:
        raise KeyError('the cfg dict must contain the key "image_branch" and "xyz_branch"')
    image_branch_type = image_branch_cfg.pop('type')
    xyz_branch_type = xyz_branch_cfg.pop('type')

    image_branch = generate_encoder_decoder(image_branch_cfg) if image_branch_type == 'EncoderDecoder' else generate_normal_model(image_branch_cfg)
    image_branch = SegmentationModel(image_branch,**image_branch_cfg)
    xyz_branch = generate_encoder_decoder(xyz_branch_cfg) if xyz_branch_type == 'EncoderDecoder' else generate_normal_model(xyz_branch_cfg)
    xyz_branch = SegmentationModel(xyz_branch,**xyz_branch_cfg)

    spatial_fusion = genrate_spatial_fusion(spatial_fusion_cfg)
    return MVMapper(image_branch, spatial_fusion, xyz_branch)

# 生成MVMapper、常规模型、EncoderDecoder模型
# 根据type是MVMapper还是EncoderDecoder生成不同的模型
def generate_model(cfg:Dict):
    cfg_ = cfg.copy()
    model_type = cfg_.pop('type')
    if model_type == 'MVMapper':
        return generate_mvmapper(cfg_)
    elif model_type == 'EncoderDecoder':
        return generate_encoder_decoder(cfg_)
    elif model_type == 'EncoderDecoderCrossAttn':
        return generate_encoder_decoder_crossattn(cfg_)
    else:
        return generate_normal_model(cfg.copy())


if __name__ == "__main__":
    from lightning.pytorch.trainer import Trainer
    from fusion_model_interface import FusionMInterface
    x=torch.ones([5,4,512,512])
    xyz=torch.ones([2,4,512,512])
    grid=torch.ones([5,128,128,2])
    batch_info=[3,2]
    from yacs_config.default_config import load_yaml
    cfg = load_yaml(r'/path/to/project/GraduateCareer\code\mvmapper_reconstruct\cfgs\model_cfg.yaml')
    model_key=('model','callback','predict','metric')
    model_cfg=dict(filter(lambda item: item[0] in model_key, cfg.items()))
    model = FusionMInterface(**model_cfg)
    trainer= Trainer(**cfg['trainer'],profiler="simple")
    # model.eval()
    # img = model(x)
    # model.set_stage('stage2')
    # xyz = model({'image':[x,xyz],'grid':grid,'batch_info':batch_info})
    # img, xyz = model({'image':[x,xyz],'grid':grid,'batch_info':batch_info})
    # print(img.shape,xyz.shape)

    for name,params in model.named_parameters():
        if params.requires_grad:
            print(name)

if __name__ == "__main222__":
    # from utils.basic_func import load_pretrain_with_checkpoint
    # model_cfg=dict(
    #     encoder=dict(
    #         type='SwinTransformer',
    #         pretrain_img_size=512,
    #         in_chans=3,
    #         window_size=7),
    #     decoder=dict(
    #         type='UPerHead',
    #         in_channels=[96, 192, 384, 768],
    #         pool_scales=(1, 2, 3, 6),
    #         out_channels=512,
    #         # dropout_ratio=0.1,
    #         norm_cfg=dict(type='BatchNorm2d', requires_grad=True),
    #         align_corners=False),
    #     spatial_fusion=dict(
    #         type='UPerFusionHead',
    #         in_channels=[96, 192, 384, 768],
    #         pool_scales=(1, 2, 3, 6),
    #         out_channels=512,
    #         norm_cfg=dict(type='BatchNorm2d', requires_grad=True),
    #         align_corners=False,
    #         mode='nearest'),
    #     num_class=1)

    # model=generate_model(model_cfg)
    # load_pretrain_with_checkpoint(model,'/path/to/user/NAS/gwx/experiment_result/20221207/2022-12-07-10-17-08-storage-upernet_swin_tiny_patch4_window7_512x512.pth','state_dict')
    # # model.load_state('/path/to/user/NAS/gwx/experiment_result/20221207/2022-12-06-19-23-55-storage-swin_tiny_patch4_window7_224_22k.pth')
    # # model.load_state('/path/to/user/NAS/gwx/experiment_result/20221207/2022-12-07-10-17-08-storage-upernet_swin_tiny_patch4_window7_512x512.pth','state_dict')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # t = model.to(device)
    # x = torch.ones([8, 3, 512, 512]).to(device)
    # out=model(x)
    # summary(t,input_size=(3,224,224),batch_size=2)
    torch.manual_seed(40)
    h,w=512,512
    from hook.basehook import HookClass
    from utils.raster_io import RasterIO    
    import cv2 
    
    ds=RasterIO('/path/to/user/NAS/gwx/code/cnncode/2.tif')
    ds_h,ds_w=ds.img_h,ds.img_w
    x=ds.read_as_array()[0:3,:,:]#.read_as_given_hw(h,w)
    x=torch.tensor(np.expand_dims(x,0))
    model=ResNet_FPN('50',3,1)
    model=model.eval()
    
    # for name,modules in model.named_modules():
    #     print(name)
    # print('_______________________________')
    # for name,modules in model.named_children():
    #     print(name)
        
    hook=HookClass(model,'stage3')
    y=model(x)

    cam=F.interpolate(hook.out_features, size=(ds_h,ds_w)).numpy().squeeze(0)
    cam=np.max(cam,axis=0)
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    x=x.numpy().squeeze(0).transpose([1,2,0])
    # 2)cam转换成与原图大小一致的彩色度(cv2.COLORMAP_HSV为彩色图的其中一种类型)
    cam_color_resnet18 = cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET)
    cam_color_resnet18=cv2.GaussianBlur(cam_color_resnet18,(9,9),0)
    # 3)合并cam和原图，并保存
    cam_resnet18 = x*0.5 + cam_color_resnet18 * 0.5
    # cv2.imshow('1',cam_color_resnet18)
    cv2.imwrite("/path/to/user/NAS/gwx/code/cnncode/2.jpg", cam_resnet18)
    # cv2.imwrite("/path/to/user/NAS/gwx/code/cnncode/cam.jpg", np.uint8(255*cam))
    # cv2.imwrite("/path/to/user/NAS/gwx/code/cnncode/x.jpg", x)
